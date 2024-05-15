__all__ = ["CovarianceNormalizationJob"]

import shutil

from sisyphus import *

Path = setup_path(__package__)

import i6_core.rasr as rasr
import i6_core.util as util


class CovarianceNormalizationJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        feature_flow,
        extra_config_estimate=None,
        extra_post_config_estimate=None,
        extra_config_normalization=None,
        extra_post_config_normalization=None,
    ):
        self.set_vis_name("Covariance Normalization")

        args = locals()
        del args["self"]

        (
            self.config_estimate,
            self.post_config_estimate,
            self.config_normalization,
            self.post_config_normalization,
        ) = CovarianceNormalizationJob.create_config(**args)
        self.feature_flow = feature_flow
        self.concurrent = crp.concurrent

        self.estimate_rqmt = {
            "time": max(crp.corpus_duration / 30.0, 0.5),
            "cpu": 1,
            "mem": 1,
        }
        self.exe = (
            crp.feature_statistics_exe
            if crp.feature_statistics_exe is not None
            else self.default_exe("feature-statistics")
        )

        self.estimate_log_file = self.log_file_output_path("estimate", crp, False)
        self.normalization_log_file = self.log_file_output_path(
            "normalization", crp, False
        )
        self.normalization_matrix = self.output_path(
            "normalization.matrix", cached=True
        )

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("estimate", resume="estimate", rqmt=self.estimate_rqmt)
        yield Task("normalization", mini_task=True)

    def create_files(self):
        self.feature_flow.write_to_file("feature.flow")
        for cmd in ["estimate", "normalization"]:
            with open("%s.config" % cmd, "wt") as f:
                config = getattr(self, "config_%s" % cmd)
                post_config = getattr(self, "post_config_%s" % cmd)
                config._update(post_config)
                f.write(repr(config))
            self.write_run_script(self.exe, "%s.config" % cmd, "%s.sh" % cmd)

    def estimate(self):
        self.run_script(1, self.estimate_log_file, "./estimate.sh")

    def normalization(self):
        self.run_script(1, self.normalization_log_file, "./normalization.sh")
        shutil.move("normalization.matrix", self.normalization_matrix.get_path())

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        if cmd == "./estimate.sh":
            util.backup_if_exists("estimate.log")
        if cmd == "./normalization.sh":
            util.backup_if_exists("normalization.log")

    @classmethod
    def create_config(
        cls,
        crp,
        feature_flow,
        extra_config_estimate=None,
        extra_post_config_estimate=None,
        extra_config_normalization=None,
        extra_post_config_normalization=None,
    ):
        # estimate
        config_estimate, post_config_estimate = rasr.build_config_from_mapping(
            crp, {"corpus": "feature-statistics.corpus"}
        )
        config_estimate.feature_statistics.action = "estimate-covariance"
        config_estimate.feature_statistics.covariance_estimator.file = (
            "xml:covariance.matrix"
        )
        config_estimate.feature_statistics.covariance_estimator.shall_normalize = True
        config_estimate.feature_statistics.covariance_estimator.feature_extraction.file = (
            "feature.flow"
        )
        config_estimate.feature_statistics.covariance_estimator.output_precision = 20

        feature_flow.apply_config(
            "feature-statistics.covariance-estimator.feature-extraction",
            config_estimate,
            post_config_estimate,
        )

        config_estimate._update(extra_config_estimate)
        post_config_estimate._update(extra_post_config_estimate)

        # normalization
        (
            config_normalization,
            post_config_normalization,
        ) = rasr.build_config_from_mapping(crp, {})
        config_normalization.feature_statistics.action = (
            "calculate-covariance-diagonal-normalization"
        )
        config_normalization.feature_statistics.covariance_diagonal_normalization.covariance_file = (
            "xml:covariance.matrix"
        )
        config_normalization.feature_statistics.covariance_diagonal_normalization.normalization_file = (
            "xml:normalization.matrix"
        )
        config_normalization.feature_statistics.covariance_diagonal_normalization.output_precision = (
            20
        )

        config_normalization._update(extra_config_normalization)
        post_config_normalization._update(extra_post_config_normalization)

        return (
            config_estimate,
            post_config_estimate,
            config_normalization,
            post_config_normalization,
        )

    @classmethod
    def hash(cls, kwargs):
        ce, pce, cn, pcn = cls.create_config(**kwargs)
        return super().hash(
            {
                "config_estimate": ce,
                "config_normalization": cn,
                "flow": kwargs["feature_flow"],
                "exe": kwargs["crp"].feature_statistics_exe,
            }
        )
