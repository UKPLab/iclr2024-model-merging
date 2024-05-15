__all__ = ["AlignmentJob", "DumpAlignmentJob", "AMScoresFromAlignmentLogJob"]

import xml.etree.ElementTree as ET
import math
import os
import shutil

from sisyphus import *

Path = setup_path(__package__)

from .flow import alignment_flow, dump_alignment_flow
import i6_core.rasr as rasr
import i6_core.util as util


class AlignmentJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        feature_flow,
        feature_scorer,
        alignment_options=None,
        word_boundaries=False,
        use_gpu=False,
        rtf=1.0,
        extra_config=None,
        extra_post_config=None,
    ):
        """
        :param rasr.crp.CommonRasrParameters crp:
        :param feature_flow:
        :param rasr.FeatureScorer feature_scorer:
        :param dict[str] alignment_options:
        :param bool word_boundaries:
        :param bool use_gpu:
        :param float rtf:
        :param extra_config:
        :param extra_post_config:
        """
        assert isinstance(feature_scorer, rasr.FeatureScorer)

        self.set_vis_name("Alignment")

        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = AlignmentJob.create_config(**kwargs)
        self.alignment_flow = AlignmentJob.create_flow(**kwargs)
        self.concurrent = crp.concurrent
        self.exe = self.select_exe(
            crp.acoustic_model_trainer_exe, "acoustic-model-trainer"
        )
        self.feature_scorer = feature_scorer
        self.use_gpu = use_gpu
        self.word_boundaries = word_boundaries

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
        if self.word_boundaries:
            self.out_single_word_boundary_caches = dict(
                (i, self.output_path("word_boundary.cache.%d" % i, cached=True))
                for i in range(1, self.concurrent + 1)
            )
            self.out_word_boundary_path = util.MultiOutputPath(
                self,
                "word_boundary.cache.$(TASK)",
                self.out_single_word_boundary_caches,
                cached=True,
            )
            self.out_word_boundary_bundle = self.output_path(
                "word_boundary.cache.bundle", cached=True
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
        if self.word_boundaries:
            util.write_paths_to_file(
                self.out_word_boundary_bundle,
                self.out_single_word_boundary_caches.values(),
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
        if self.word_boundaries:
            shutil.move(
                "word_boundary.cache.%d" % task_id,
                self.out_single_word_boundary_caches[task_id].get_path(),
            )

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("alignment.log.%d" % task_id)
        util.delete_if_exists("alignment.cache.%d" % task_id)
        if self.word_boundaries:
            util.delete_if_zero("word_boundary.cache.%d" % task_id)

    @classmethod
    def create_config(
        cls,
        crp,
        feature_flow,
        feature_scorer,
        alignment_options,
        word_boundaries,
        extra_config,
        extra_post_config,
        **kwargs,
    ):
        """
        :param rasr.crp.CommonRasrParameters crp:
        :param feature_flow:
        :param rasr.FeatureScorer feature_scorer:
        :param dict[str] alignment_options:
        :param bool word_boundaries:
        :param extra_config:
        :param extra_post_config:
        :return: config, post_config
        :rtype: (rasr.RasrConfig, rasr.RasrConfig)
        """
        alignment_flow = cls.create_flow(feature_flow)

        # TODO: think about mode
        alignopt = {
            "increase-pruning-until-no-score-difference": True,
            "min-acoustic-pruning": 500,
            "max-acoustic-pruning": 4000,
            "acoustic-pruning-increment-factor": 2,
        }
        if alignment_options is not None:
            alignopt.update(alignment_options)

        mapping = {
            "corpus": "acoustic-model-trainer.corpus",
            "lexicon": [],
            "acoustic_model": [],
        }

        # acoustic model + lexicon for the flow nodes
        for node in alignment_flow.get_node_names_by_filter("speech-alignment"):
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

        # alignment options for the flow nodes
        for node in alignment_flow.get_node_names_by_filter("speech-alignment"):
            node_config = config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction[
                node
            ]

            node_config.aligner = rasr.RasrConfig()
            for k, v in alignopt.items():
                node_config.aligner[k] = v
            feature_scorer.apply_config(
                "model-combination.acoustic-model.mixture-set", node_config, node_config
            )

            if word_boundaries:
                node_config.store_lattices = True
                node_config.lattice_archive.path = "word_boundary.cache.$(TASK)"

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
    def create_flow(cls, feature_flow, **kwargs):
        return alignment_flow(feature_flow, "alignment.cache.$(TASK)")

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


class DumpAlignmentJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        feature_flow,
        original_alignment,
        extra_config=None,
        extra_post_config=None,
    ):
        self.set_vis_name("Dump Alignment")

        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = DumpAlignmentJob.create_config(**kwargs)
        self.dump_flow = DumpAlignmentJob.create_flow(**kwargs)
        self.exe = self.select_exe(
            crp.acoustic_model_trainer_exe, "acoustic-model-trainer"
        )
        self.concurrent = crp.concurrent

        self.out_log_file = self.log_file_output_path("dump", crp, True)
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
            "time": max(crp.corpus_duration / (50.0 * crp.concurrent), 0.5),
            "cpu": 1,
            "mem": 1,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmt, args=range(1, self.concurrent + 1))

    def create_files(self):
        self.write_config(self.config, self.post_config, "dump.config")
        self.dump_flow.write_to_file("dump.flow")
        util.write_paths_to_file(
            self.out_alignment_bundle, self.out_single_alignment_caches.values()
        )
        self.write_run_script(self.exe, "dump.config")

    def run(self, task_id):
        self.run_script(task_id, self.out_log_file[task_id])
        shutil.move(
            "alignment.cache.%d" % task_id,
            self.out_single_alignment_caches[task_id].get_path(),
        )

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("dump.log.%d" % task_id)
        util.delete_if_zero("alignment.cache.%d" % task_id)

    @classmethod
    def create_config(cls, crp, extra_config, extra_post_config, **kwargs):
        dump_flow = cls.create_flow(**kwargs)

        mapping = {
            "corpus": "acoustic-model-trainer.corpus",
            "lexicon": [],
            "acoustic_model": [],
        }

        # acoustic model + lexicon for the flow nodes
        for node in dump_flow.get_node_names_by_filter("speech-alignment-dump"):
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

        config.acoustic_model_trainer.action = "dry"
        config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction.file = (
            "dump.flow"
        )
        post_config["*"].allow_overwrite = True

        dump_flow.apply_config(
            "acoustic-model-trainer.aligning-feature-extractor.feature-extraction",
            config,
            post_config,
        )

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def create_flow(cls, feature_flow, original_alignment, **kwargs):
        return dump_alignment_flow(
            feature_flow, original_alignment, "alignment.cache.$(TASK)"
        )

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        dump_flow = cls.create_flow(**kwargs)
        return super().hash(
            {
                "config": config,
                "dump_flow": dump_flow,
                "exe": kwargs["crp"].acoustic_model_trainer_exe,
            }
        )


class AMScoresFromAlignmentLogJob(Job):
    def __init__(self, logs):
        self.logs = logs
        self.out_report = self.output_path("report.txt")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        with open(self.out_report.get_path(), "wt") as f:
            for log in self.logs:
                if isinstance(log, dict):
                    log = log.values()
                else:
                    log = [log]

                total_score = 0.0
                total_frames = 1
                for l in log:
                    with util.uopen(tk.uncached_path(l), "rt") as infile:
                        tree = ET.parse(infile)
                    for e in tree.findall(".//alignment-statistics"):
                        total_frames += int(e.find("./frames").text)
                        total_score += float(e.find("./score/total").text)

                f.write("%f\n" % (total_score / total_frames))
