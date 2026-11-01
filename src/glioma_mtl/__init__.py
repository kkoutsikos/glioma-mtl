"""SSL pretraining, segmentation and multi-task classification of diffuse gliomas (UCSF-PDGM)."""

# Import torch before pandas/scikit-learn: on some Windows setups loading their OpenMP/MKL DLLs first
# makes the later torch/MONAI import crash with an access violation.
import torch  # noqa: F401

__version__ = "0.1.0"
