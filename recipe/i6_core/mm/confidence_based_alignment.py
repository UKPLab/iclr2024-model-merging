__all__ = ["ConfidenceBasedAlignmentJob"]

import math
import os
import shutil

from sisyphus import *

Path = setup_path(__package__)

from .flow import confidence_based_alignment_flow
import i6_core.rasr as rasr
import i6_core.util as util


class ConfidenceBasedAlignmentJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        feature_flow,
        feature_scorer,
        lattice_cache,
        *,
        global_scale=1.0,
        confidence_threshold=0.75,
        weight_scale=1.0,
        ref_alignment_path=None,
        use_gpu=False,
        rtf=0.5,
        extra_config=None,
        extra_post_config=None,
    ):
        assert isinstance(feature_scorer, rasr.FeatureScorer)

        self.set_vis_name("Confidence-based alignment")

        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = ConfidenceBasedAlignmentJob.create_config(
            **kwargs
        )
        self.alignment_flow = ConfidenceBasedAlignmentJob.create_flow(**kwargs)
        self.concurrent = crp.concurrent
        self.exe = self.select_exe(
            crp.acoustic_model_trainer_exe, "acoustic-model-trainer"
        )
        self.feature_scorer = feature_scorer
        self.use_gpu = use_gpu

        self.out_log_file = self.log_file_output_path("alignment", crp, True)
        self.out_single_alignment_caches = dict(
            (i, self.output_path("alignment.cache.%d" % i, cached=True))
            for i in range(1, self.concurrent + 1)
        )
        self.out_alignment_path = util.MultiOutputPath(
            self,
            "alignment.cache.$(TASK)",
            self.out_single_alignment_caches,
            cached=True,
        )
        self.out_alignment_bundle = self.output_path(
            "alignment.cache.bundle", cached=True
        )

        self.rqmt = {
            "time": max(rtf * crp.corpus_duration / crp.concurrent, 0.5),
            "cpu": 1,
            "gpu": 1 if self.use_gpu else 0,
            "mem": 2,
        }

    def tasks(self):
        rqmt = self.rqmt.copy()
        if isinstance(self.feature_scorer, rasr.GMMFeatureScorer):
            mixture_size = os.stat(
                tk.uncached_path(self.feature_scorer.config["file"])
            ).st_size / (1024.0**2)
            rqmt["mem"] += int(math.ceil((mixture_size - 200.0) / 750.0))

        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=rqmt, args=range(1, self.concurrent + 1))

    def create_files(self):
        self.write_config(self.config, self.post_config, "alignment.config")
        self.alignment_flow.write_to_file("alignment.flow")
        util.write_paths_to_file(
            self.out_alignment_bundle, self.out_single_alignment_caches.values()
        )
        extra_code = (
            ":${{THEANO_FLAGS:="
            "}}\n"
            'export THEANO_FLAGS="$THEANO_FLAGS,device={0},force_device=True"\n'
            'export TF_DEVICE="{0}"'.format("gpu" if self.use_gpu else "cpu")
        )
        self.write_run_script(self.exe, "alignment.config", extra_code=extra_code)

    def run(self, task_id):
        self.run_script(task_id, self.out_log_file[task_id])
        shutil.move(
            "alignment.cache.%d" % task_id,
            self.out_single_alignment_caches[task_id].get_path(),
        )

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("alignment.log.%d" % task_id)
        util.delete_if_exists("alignment.cache.%d" % task_id)

    @classmethod
    def create_config(
        cls,
        crp,
        feature_flow,
        feature_scorer,
        lattice_cache,
        global_scale,
        confidence_threshold,
        weight_scale,
        ref_alignment_path,
        extra_config,
        extra_post_config,
        **kwargs,
    ):
        alignment_flow = cls.create_flow(
            feature_flow,
            lattice_cache,
            global_scale,
            confidence_threshold,
            weight_scale,
            ref_alignment_path,
        )
        mapping = {
            "corpus": "acoustic-model-trainer.corpus",
            "lexicon": [],
            "acoustic_model": [],
        }

        # acoustic model + lexicon for the flow nodes
        for node_type in [
            "model-combination",
            "alignment-weights-by-tied-state-alignment-weights",
        ]:
            for node in alignment_flow.get_node_names_by_filter(node_type):
                mapping["lexicon"].append(
                    "acoustic-model-trainer.aligning-feature-extractor.feature-extraction.%s.model-combination.lexicon"
                    % node
                )
                mapping["acoustic_model"].append(
                    "acoustic-model-trainer.aligning-feature-extractor.feature-extraction.%s.model-combination.acoustic-model"
                    % node
                )

        config, post_config = rasr.build_config_from_mapping(
            crp, mapping, parallelize=True
        )

        for node_type in [
            "model-combination",
            "alignment-weights-by-tied-state-alignment-weights",
        ]:
            for node in alignment_flow.get_node_names_by_filter(node_type):
                node_config = config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction[
                    node
                ]
                node_post_config = post_config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction[
                    node
                ]
                feature_scorer.apply_config(
                    "model-combination.acoustic-model.mixture-set",
                    node_config,
                    node_post_config,
                )

        alignment_flow.apply_config(
            "acoustic-model-trainer.aligning-feature-extractor.feature-extraction",
            config,
            post_config,
        )

        config.action = "dry"
        config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction.file = (
            "alignment.flow"
        )
        post_config["*"].allow_overwrite = True

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def create_flow(
        cls,
        feature_flow,
        lattice_cache,
        global_scale,
        confidence_threshold,
        weight_scale,
        ref_alignment_path,
        **kwargs,
    ):
        return confidence_based_alignment_flow(
            feature_flow,
            lattice_cache,
            "alignment.cache.$(TASK)",
            global_scale,
            confidence_threshold,
            weight_scale,
            ref_alignment_path,
        )

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        alignment_flow = cls.create_flow(**kwargs)
        return super().hash(
            {
                "config": config,
                "alignment_flow": alignment_flow,
                "exe": kwargs["crp"].acoustic_model_trainer_exe,
            }
        )
