__all__ = ["ScoreFeaturesWithWarpingFactorsJob", "EstimateWarpingMixturesJob"]

import collections
import copy
import os
import xml.etree.ElementTree as ET

from sisyphus import *

Path = setup_path(__package__)

import i6_core.mm as mm
import i6_core.rasr as rasr
import i6_core.util as util

from .flow import *


class ScoreFeaturesWithWarpingFactorsJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        feature_flow,
        feature_scorer,
        alignment,
        alphas=[f / 100.0 for f in range(88, 123, 2)],
        omega=0.875,
        filterbank_node="filterbank",
        extra_config=None,
        extra_post_config=None,
    ):
        kwargs = locals()
        del kwargs["self"]

        self.concurrent = crp.concurrent
        (
            self.config,
            self.post_config,
        ) = ScoreFeaturesWithWarpingFactorsJob.create_config(**kwargs)
        self.exe = self.select_exe(
            crp.acoustic_model_trainer_exe, "acoustic-model-trainer"
        )
        self.flow = ScoreFeaturesWithWarpingFactorsJob.create_flow(**kwargs)
        self.alphas = copy.copy(alphas)

        self.log_file = {
            idx: self.log_file_output_path("score_%d" % idx, crp, True)
            for idx in range(len(self.alphas))
        }
        self.alphas_file = self.output_path("alphas.xml")
        self.warping_map = self.output_path("warping_map.xml")

        self.rqmt = {
            "time": max(crp.corpus_duration / (50.0 * crp.concurrent), 0.5),
            "cpu": 1,
            "mem": 1,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task(
            "run", resume="run", rqmt=self.rqmt, args=range(1, self.concurrent + 1)
        )
        yield Task("build_map", mini_task=True)

    def create_files(self):
        self.write_config(self.config, self.post_config, "score.config")
        self.flow.write_to_file("feature_and_alignment.flow")
        self.write_run_script(self.exe, "score.config")

    def run(self, task_id):
        for idx, wf in enumerate(self.alphas):
            self.run_script(
                task_id,
                self.log_file[idx][task_id],
                args=[
                    "--*.WARPING-FACTOR=%s" % wf,
                    "--*.SCORE-LOGFILE=scores_%d_%d.log" % (idx, task_id),
                ],
            )

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        for idx in range(len(self.alphas)):
            util.backup_if_exists("score_%d.log.%d" % (idx, task_id))

    def build_map(self):
        segment_scores = collections.defaultdict(list)
        for idx, wf in enumerate(self.alphas):
            for task_id in range(1, self.concurrent + 1):
                tree = ET.parse("scores_%d_%d.log" % (idx, task_id))
                for seg in tree.findall(".//score-accumulator"):
                    segment_scores[seg.attrib["corpus-key"]].append(
                        float(seg.find(".//weighted-sum-of-scores").text)
                    )

        used_alphas = set()
        with open(self.warping_map.get_path(), "wt") as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write("<coprus-key-map>\n")

            for seg in sorted(segment_scores):
                scores = segment_scores[seg]
                argmin = min(enumerate(scores), key=lambda e: e[1])[0]
                f.write(
                    '  <map-item key="%s" value="%s"/>\n' % (seg, self.alphas[argmin])
                )
                used_alphas.add(self.alphas[argmin])

            f.write("</coprus-key-map>")

        with open(self.alphas_file.get_path(), "wt") as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n<vector-string>\n')
            for factor in sorted(used_alphas):
                f.write("  %s\n" % factor)
            f.write("</vector-string>")

    @classmethod
    def create_config(
        cls,
        crp,
        feature_flow,
        alignment,
        feature_scorer,
        omega,
        filterbank_node,
        extra_config,
        extra_post_config,
        **kwargs,
    ):
        feature_flow = cls.create_flow(feature_flow, alignment, filterbank_node)

        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "acoustic_model": "acoustic-model-trainer.feature-scorer.acoustic-model",
                "corpus": "acoustic-model-trainer.corpus",
                "lexicon": "acoustic-model-trainer.feature-scorer.lexicon",
            },
            parallelize=True,
        )

        config.action = "score-features"

        post_config["*"].segment_map_channel.append = False
        post_config["*"].segment_map_channel.compressed = False
        post_config["*"].segment_map_channel.encoding = "UTF-8"
        post_config["*"].segment_map_channel.file = "$(SCORE-LOGFILE)"
        post_config["*"].segment_map_channel.unbuffered = False

        config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction.file = (
            "feature_and_alignment.flow"
        )
        config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction.warping_alpha = (
            "$(WARPING-FACTOR)"
        )
        config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction.warping_omega = (
            omega
        )
        config.acoustic_model_trainer.feature_scorer.corpus_key.template = "<segment>"
        post_config.acoustic_model_trainer.feature_scorer.output.channel = (
            "segment-map-channel"
        )

        feature_flow.apply_config(
            "acoustic-model-trainer.aligning-feature-extractor.feature-extraction",
            config,
            post_config,
        )
        feature_scorer.apply_config(
            "acoustic-model-trainer.feature-scorer.acoustic-model.mixture-set",
            config,
            post_config,
        )

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def create_flow(cls, feature_flow, alignment, filterbank_node, **kwargs):
        feature_flow = add_static_warping_to_filterbank_flow(
            feature_net=feature_flow, node_name=filterbank_node
        )
        return mm.cached_alignment_flow(feature_flow, alignment)

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        return super().hash(
            {
                "config": config,
                "feature_flow": ScoreFeaturesWithWarpingFactorsJob.create_flow(
                    **kwargs
                ),
                "exe": kwargs["crp"].acoustic_model_trainer_exe,
            }
        )


class EstimateWarpingMixturesJob(mm.MergeMixturesJob):
    def __init__(
        self,
        crp,
        old_mixtures,
        feature_flow,
        warping_map,
        warping_factors,
        split_first=True,
        keep_accumulators=False,
        extra_warping_args=None,
        extra_merge_args=None,
        extra_config=None,
        extra_post_config=None,
    ):
        self.set_vis_name(
            "Split Warping Mixtures" if split_first else "Accumulate Warping Mixtures"
        )

        kwargs = locals()
        del kwargs["self"]
        super().__init__(**mm.EstimateMixturesJob.merge_args(**kwargs))

        self.config, self.post_config = EstimateWarpingMixturesJob.create_config(
            **kwargs
        )
        self.label_flow = EstimateWarpingMixturesJob.create_flow(**kwargs)
        self.exe = self.select_exe(
            crp.acoustic_model_trainer_exe, "acoustic-model-trainer"
        )
        self.split_first = split_first
        self.keep_accumulators = keep_accumulators
        self.concurrent = crp.concurrent

        self._old_mixtures = old_mixtures

        self.log_file = self.log_file_output_path("accumulate", crp, True)

        self.accumulate_rqmt = {
            "time": max(crp.corpus_duration / 20, 0.5),
            "cpu": 1,
            "mem": 2,
        }

    def tasks(self):
        rqmt = self.accumulate_rqmt.copy()
        try:
            mixture_size = os.stat(tk.uncached_path(self._old_mixtures)).st_size / (
                1024.0**2
            )
            rqmt["mem"] += 2 if mixture_size > 500.0 else 0
        except OSError as e:
            if e.errno != 2:  # file does not exist
                raise

        yield Task("create_files", mini_task=True)
        yield Task(
            "accumulate",
            resume="accumulate",
            rqmt=rqmt,
            args=range(1, self.concurrent + 1),
        )
        yield from super().tasks()
        if not self.keep_accumulators:
            yield Task("delete_accumulators", mini_task=True)

    def create_files(self):
        self.write_config(self.config, self.post_config, "accumulate-mixtures.config")
        self.label_flow.write_to_file("label.flow")
        self.write_run_script(self.exe, "accumulate-mixtures.config", "accumulate.sh")

    def accumulate(self, task_id):
        self.run_script(task_id, self.log_file[task_id], "./accumulate.sh")

    def delete_accumulators(self):
        for i in range(1, self.concurrent + 1):
            if os.path.exists("am.acc.%d" % i):
                os.remove("am.acc.%d" % i)

    def cleanup_before_run(self, cmd, retry, *args):
        if cmd == self.merge_exe:
            super().cleanup_before_run(cmd, retry, *args)
        else:
            task_id = args[0]
            util.backup_if_exists("accumulate.log.%d" % task_id)

    @classmethod
    def create_config(
        cls,
        crp,
        old_mixtures,
        feature_flow,
        warping_map,
        warping_factors,
        split_first,
        extra_warping_args,
        extra_config,
        extra_post_config,
        **kwargs,
    ):
        label_flow = cls.create_flow(feature_flow, warping_map, extra_warping_args)

        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "corpus": "acoustic-model-trainer.corpus",
                "acoustic_model": "acoustic-model-trainer.mixture-set-trainer.acoustic-model",
            },
            parallelize=True,
        )

        config.acoustic_model_trainer.action = "accumulate-mixture-set-text-independent"
        config.acoustic_model_trainer.labeling.feature_extraction.file = "label.flow"
        config.acoustic_model_trainer.labeling.labels = warping_factors
        config.acoustic_model_trainer.mixture_set_trainer.split_first = split_first
        config.acoustic_model_trainer.mixture_set_trainer.old_mixture_set_file = (
            old_mixtures
        )
        config.acoustic_model_trainer.mixture_set_trainer.new_mixture_set_file = (
            "am.acc.$(TASK)"
        )

        label_flow.apply_config(
            "acoustic-model-trainer.labeling.feature-extraction", config, post_config
        )

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def create_flow(cls, feature_flow, warping_map, extra_warping_args, **kwargs):
        kwargs = extra_warping_args if extra_warping_args is not None else {}
        return label_features_with_map_flow(feature_flow, warping_map, **kwargs)

    @classmethod
    def merge_args(cls, crp, extra_merge_args, **kwargs):
        merge_args = {
            "crp": crp,
            "mixtures_to_combine": [
                "am.acc.%d" % i for i in range(1, crp.concurrent + 1)
            ],
            "combine_per_step": 2,
            "estimator": "maximum-likelihood",
            "extra_config": None,
            "extra_post_config": None,
        }
        if extra_merge_args is not None:
            merge_args.update(extra_merge_args)
        return merge_args

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        alignment_flow = cls.create_flow(**kwargs)
        split = "split." if kwargs["split_first"] else "accumulate."
        return split + Job.hash(
            {
                "config": config,
                "alignment_flow": alignment_flow,
                "exe": kwargs["crp"].acoustic_model_trainer_exe,
                "merge_hash": mm.MergeMixturesJob.hash(cls.merge_args(**kwargs)),
            }
        )
