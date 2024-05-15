__all__ = ["CreateLmImageJob"]

import shutil
import logging

from sisyphus import *
from i6_core import rasr
from i6_core import util

Path = setup_path(__package__)


class CreateLmImageJob(rasr.RasrCommand, Job):
    """
    pre-compute LM image without generating global cache
    """

    def __init__(
        self,
        crp,
        extra_config=None,
        extra_post_config=None,
        encoding="utf-8",
        mem=2,
    ):
        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = CreateLmImageJob.create_config(**kwargs)
        self.exe = self.select_exe(crp.lm_util_exe, "lm-util")

        self.log_file = self.log_file_output_path("lm_image", crp, False)
        self.out_image = self.output_path("lm.image", cached=True)

        self.rqmt = {"time": 1, "cpu": 1, "mem": mem}

    def tasks(self):
        yield Task("create_files", resume="create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def create_files(self):
        self.write_config(self.config, self.post_config, "lm_image.config")
        extra_code = (
            ":${{THEANO_FLAGS:="
            "}}\n"
            'export THEANO_FLAGS="$THEANO_FLAGS,device={0},force_device=True"\n'
            'export TF_DEVICE="{0}"'.format(
                "gpu" if self.rqmt.get("gpu", 0) > 0 else "cpu"
            )
        )
        self.write_run_script(self.exe, "lm_image.config", extra_code=extra_code)

    def run(self):
        self.run_script(1, self.log_file)
        shutil.move("lm.image", self.out_image.get_path())

    def cleanup_before_run(self, cmd, retry, *args):
        util.backup_if_exists("lm_image.log")

    @classmethod
    def create_config(
        cls,
        crp,
        extra_config,
        extra_post_config,
        encoding,
        **kwargs,
    ):
        config, post_config = rasr.build_config_from_mapping(
            crp, {"lexicon": "lm-util.lexicon", "language_model": "lm-util.lm"}
        )
        del (
            config.lm_util.lm.scale
        )  # scale not considered here, delete to remove ambiguity

        assert config.lm_util.lm.type == "ARPA"

        if "image" in post_config.lm_util.lm:
            logging.warning(
                "The LM image already exists, but a new one will be recreated."
            )
            del post_config.lm_util.lm.image

        config.lm_util.action = "load-lm"
        config.lm_util.lm.image = "lm.image"
        config.lm_util.encoding = encoding
        config.lm_util.batch_size = 100

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        return super().hash({"config": config, "exe": kwargs["crp"].lm_util_exe})
