The `tests` package contains pytest-compatible python files for testing.

Currently the test-types are limited to `job_tests`, which test a specific input/output combination for certain jobs.

To run local testing, call: `python3 -m pytest i6_core/tests/`from one folder above i6_core. Make sure that `sisyphus`is part of your `PYTHONPATH`, otherwise exectution will crash.