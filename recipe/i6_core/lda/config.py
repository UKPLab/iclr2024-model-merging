__all__ = [
    "build_eigenvalue_problem_config",
    "build_generalized_eigenvalue_problem_config",
]

from i6_core.rasr import ConfigBuilder

build_eigenvalue_problem_config = ConfigBuilder(
    {
        "type": "symmetric",
        "eigenvector-normalization-type": "unity-length",
        "eigenvector-normalization-tolerance": 0.0,
        "eigenvector-relative-imaginary-maximum": 1.0,
        "verification-tolerance": 1e16,
        "eigenvalue-sort-type": "decreasing",
        "eigenvalue-lower-bound": -1.7976931348623157e308,
        "eigenvalue-upper-bound": -1.7976931348623157e308,  # rasr does special handling if lower = upper bound = min double
        "driver": "relatively-robust",
        "support-eigenvectors": False,
    }
)

build_generalized_eigenvalue_problem_config = ConfigBuilder(
    {
        "type": "general",
        "eigenvector-normalization-type": "unity-length",
        "eigenvector-normalization-tolerance": 0.0,
        "eigenvector-relative-imaginary-maximum": 1e-50,
        "verification-tolerance": 1e16,
        "eigenvalue-sort-type": "decreasing",
        "normalize-eigenvectors-using-B": True,
        "eigenvalue-lower-bound": -1.7976931348623157e308,
        "eigenvalue-upper-bound": -1.7976931348623157e308,  # rasr does special handling if lower = upper bound = min double
        "driver": "expert",
        "support-eigenvectors": False,
        "relative-alpha-minimum": 2.2204460492503131e-16,
        "relative-beta-minimum": 2.2204460492503131e-16,
        "balancing": "permute-and-scale",
    }
)
