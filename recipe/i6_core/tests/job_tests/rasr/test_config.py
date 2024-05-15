import os
from sisyphus import Path
import tempfile

from i6_core.rasr.config import RasrConfig, WriteRasrConfigJob


TEST_CONFIG = """[*]
dummy-variable       = 42
other-dummy-variable = 24
"""


def test_write_rasr_config():
    """
    This test can be used to test the writing of different variable types into a rasr config
    and check for correct serialization. Only dummy example for now.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        rasr_config = RasrConfig()
        rasr_config["*"].dummy_variable = 42

        post_rasr_config = RasrConfig()
        post_rasr_config["*"].other_dummy_variable = 24

        write_rasr_config_job = WriteRasrConfigJob(rasr_config, post_rasr_config)
        write_rasr_config_job.out_config = Path(os.path.join(tmpdir, "rasr.config"))
        write_rasr_config_job.run()

        with open(write_rasr_config_job.out_config) as config_file:
            for i, (source_line, reference_line) in enumerate(
                zip(config_file.readlines(), TEST_CONFIG.split("\n"))
            ):
                assert (
                    source_line.strip() == reference_line.strip()
                ), "line mismatch in %i:\n%s vs %s" % (i, source_line, reference_line)
