"""deconv_attributor: Minimal proof-of-concept for separating cell-type shifts vs. expression changes."""

from .pipeline import run_attribution_pipeline
from .reference import build_reference, load_reference, save_reference
from .solver import AttributionOutput, attribute_changes

__all__ = [
    "AttributionOutput",
    "attribute_changes",
    "build_reference",
    "load_reference",
    "run_attribution_pipeline",
    "save_reference",
]
