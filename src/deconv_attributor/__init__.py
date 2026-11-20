"""deconv_attributor: Minimal proof-of-concept for separating cell-type shifts vs. expression changes."""

from .benchmark import BenchmarkResult, run_pseudobulk_benchmark
from .reference import Reference, build_reference, load_reference, save_reference
from .solver import AttributionOutput, attribute_changes, to_cp10k

__all__ = [
    "AttributionOutput",
    "BenchmarkResult",
    "Reference",
    "attribute_changes",
    "build_reference",
    "load_reference",
    "run_pseudobulk_benchmark",
    "save_reference",
    "to_cp10k",
]
