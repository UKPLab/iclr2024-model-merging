__all__ = ["EstimateCMLLRJob"]

import os
import shutil
import tempfile

from sisyphus import *

Path = setup_path(__package__)

from i6_core.mm.flow import cached_alignment_flow
import i6_core.rasr as rasr
import i6_core.util as util


class EstimateCMLLRJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        feature_flow,
        mixtures,
        alignment,
        cluster_map,
        num_clusters,
        estimation_iter=50,
        min_observation_weight=None,
        optimization_criterion="mmi-prime",
        extra_config=None,
        extra_post_config=None,
    ):
        self.set_vis_name("Estimate CMLLR matrices")

        kwargs = locals()
        del kwargs["self"]

        self.concurrent = crp.concurrent
        self.config, self.post_config = EstimateCMLLRJob.create_config(**kwargs)
        self.exe = self.select_exe(
            crp.acoustic_model_trainer_exe, "acoustic-model-trainer"
        )
        self.feature_flow = EstimateCMLLRJob.create_flow(**kwargs)
        self.num_clusters = num_clusters
        self.log_suffix = ".gz" if crp.compress_log_file else ""

        self.log_dir = self.output_path("logs", True)
        self.transforms = self.output_path("transforms", True)

        self.rqmt = {
            "time": max(crp.corpus_duration / (50 * self.concurrent), 0.5),
            "cpu": 1,
            "mem": 2,
        }

    def tasks(self):
        num_clusters = util.get_val(self.num_clusters)
        yield Task("create_files", mini_task=True)
        yield Task(
            "run",
            resume="run",
            rqmt=self.rqmt,
            args=range(1, num_clusters + 1),
            parallel=self.concurrent,
        )
        yield Task("move_transforms", mini_task=True)

    def create_files(self):
        self.feature_flow.write_to_file("alignment_and_feature.flow")
        self.write_config(self.config, self.post_config, "estimate-cmllr.config")
        self.write_run_script(self.exe, "estimate-cmllr.config")

    def run(self, task_id):
        log_file = os.path.join(
            self.log_dir.get_path(), "estimate-cmllr.%d%s" % (task_id, self.log_suffix)
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = [
                "--affine-feature-transform-estimator.accumulator-cache.directory=%s"
                % tmp_dir
            ]
            self.run_script(task_id, log_file, args=args)
            try:
                os.mkdir("accumulator-cache")
            except FileExistsError:
                pass
            for f in os.listdir(tmp_dir):
                shutil.move(
                    os.path.join(tmp_dir, f), os.path.join("accumulator-cache", f)
                )

    def move_transforms(self):
        for f in os.listdir("transforms"):
            shutil.move(os.path.join("transforms", f), self.transforms.get_path())

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("estimate-cmllr.log.%d" % task_id)

    @classmethod
    def create_config(
        cls,
        crp,
        feature_flow,
        mixtures,
        alignment,
        cluster_map,
        estimation_iter,
        min_observation_weight,
        optimization_criterion,
        extra_config,
        extra_post_config,
        **kwargs,
    ):
        feature_flow = cls.create_flow(feature_flow, alignment)

        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "acoustic_model": "acoustic-model-trainer.affine-feature-transform-estimator.acoustic-model",
                "corpus": "acoustic-model-trainer.corpus",
                "lexicon": "acoustic-model-trainer.affine-feature-transform-estimator.lexicon",
            },
            parallelize=True,
        )
        config.acoustic_model_trainer.action = "estimate-affine-feature-transform"

        afte = config.acoustic_model_trainer.affine_feature_transform_estimator
        afte.accumulator_cache.size = 1
        afte.accumulator_cache.directory = "accumulator-cache"
        afte.accumulator_cache.dump = False
        afte.acoustic_model.mixture_set.feature_scorer_type = "SIMD-diagonal-maximum"
        afte.acoustic_model.mixture_set.file = mixtures
        afte.corpus_key.template = "<segment>"
        afte.corpus_key_map = cluster_map
        afte.estimation_iterations = estimation_iter
        afte.feature_scorer_type = "SIMD-diagonal-maximum"
        afte.optimization_criterion = optimization_criterion
        afte.transform_directory = "transforms"
        if min_observation_weight is not None:
            afte.min_observation_weight = min_observation_weight

        config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction.file = (
            "alignment_and_feature.flow"
        )

        feature_flow.apply_config(
            "acoustic-model-trainer.aligning-feature-extractor.feature-extraction",
            config,
            post_config,
        )

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def create_flow(cls, feature_flow, alignment, **kwargs):
        return cached_alignment_flow(feature_flow, alignment)

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        flow = cls.create_flow(**kwargs)
        return super().hash(
            {
                "config": config,
                "flow": flow,
                "exe": kwargs["crp"].acoustic_model_trainer_exe,
            }
        )
