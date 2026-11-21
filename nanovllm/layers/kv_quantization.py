"""Eager INT8 reference storage, with a separate scale per token and KV head."""
import torch


def write_int8_cache(values, cache, scales, slots):
    values = values.float()
    scale = values.abs().amax(dim=-1, keepdim=True) / 127
    scale = torch.where(scale > 0, scale, 1)
    quantized = (values / scale).round().clamp(-127, 127).to(torch.int8)
    cache.flatten(0, 1).index_copy_(0, slots, quantized)
    scales.flatten(0, 1).index_copy_(0, slots, scale)


def read_int8_cache(cache, scales, slots, dtype):
    values = cache.flatten(0, 1).index_select(0, slots).float()
    return (values * scales.flatten(0, 1).index_select(0, slots)).to(dtype)


def packed_cache_slots(block_tables, cumulative_lengths, block_size, total_tokens):
    """Map packed logical positions to slots listed in each request's page table."""
    packed = torch.arange(total_tokens, device=block_tables.device)
    rows = torch.searchsorted(cumulative_lengths[1:], packed, right=True)
    position = packed - cumulative_lengths[rows]
    return block_tables[rows, position // block_size].long() * block_size + position % block_size
