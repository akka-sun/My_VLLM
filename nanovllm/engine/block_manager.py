from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks < 1 or block_size < 1:
            raise ValueError("KV cache block count and size must be positive")
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def get_computed_blocks(self, seq: Sequence) -> list[int]:
        """Find reusable full prefix pages, leaving at least one query token.

        This is an advisory lookup; allocate_slots pins these pages atomically
        before evicting anything from the free-page LRU.
        """
        h = -1
        hits = []
        for i in range((seq.num_tokens - 1) // self.block_size):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            hits.append(block_id)
        return hits

    def allocate_slots(self, seq: Sequence, num_new_tokens: int,
                       computed_blocks: list[int] | None = None, *,
                       full_sequence_must_fit: bool = False,
                       lookahead: int = 0) -> list[int] | None:
        """Allocate only through this step's end; failure has no side effects.

        The full-sequence gate checks admission capacity without allocating
        future prompt pages. Prefix pages may be active/shared or free/cached.
        """
        if num_new_tokens < 1:
            raise ValueError("num_new_tokens must be positive")
        computed_blocks = computed_blocks or []
        if seq.block_table and computed_blocks:
            raise ValueError("prefix adoption requires an unallocated sequence")
        computed = len(computed_blocks) * self.block_size if computed_blocks else seq.num_computed_tokens
        end = computed + num_new_tokens
        if lookahead < 0 or end > seq.num_tokens + lookahead:
            raise ValueError("scheduled tokens exceed sequence length")
        existing = len(seq.block_table) + len(computed_blocks)
        num_new_blocks = max(0, (end + self.block_size - 1) // self.block_size - existing)
        free_hits = sum(self.blocks[i].ref_count == 0 for i in computed_blocks)
        required = num_new_blocks
        if full_sequence_must_fit:
            required = max(required, (seq.num_tokens + self.block_size - 1) // self.block_size - existing)
        if free_hits + required > len(self.free_block_ids):
            return None
        # Pin ALL hits before allocating: a free hit might be first in the LRU.
        for block_id in computed_blocks:
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        new_blocks = [self._allocate_block() for _ in range(num_new_blocks)]
        seq.block_table.extend(new_blocks)
        seq.num_computed_tokens = computed
        return new_blocks

    def truncate(self, seq: Sequence):
        """Discard lookahead pages. Stale KV inside the tail is unreachable.

        Only verified full pages are ever hashed, so truncation cannot leave
        speculative token hashes available to another request.
        """
        keep = (seq.num_computed_tokens + self.block_size - 1) // self.block_size
        while len(seq.block_table) > keep:
            block_id = seq.block_table.pop()
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_computed_tokens = 0
        seq.block_table.clear()

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
