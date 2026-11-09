"""Load engine logic without importing the optional CUDA model stack."""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("nanovllm")
package.__path__ = [str(ROOT / "nanovllm")]
sys.modules.setdefault("nanovllm", package)

# Only model execution is replaced; scheduling and sequence state remain real.
runner = types.ModuleType("nanovllm.engine.model_runner")
runner.ModelRunner = object
sys.modules.setdefault("nanovllm.engine.model_runner", runner)
