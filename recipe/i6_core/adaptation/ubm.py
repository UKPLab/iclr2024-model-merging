__all__ = [
    "UbmLabelMapJob",
    "EstimateUniversalBackgroundMixturesJob",
    "TrainUniversalBackgroundModelSequence",
]

import os

from sisyphus import *

Path = setup_path(__package__)

import i6_core.mm as mm
import i6_core.rasr as rasr
import i6_core.util as util


def label_features_with_map_flow(
    feature_net, map_file, map_key="$(id)", default_output=0.0
):
    """
    augments a feature-net to outputs network:labels based on coprus-key-map
    :param feature_net: base feature-net
    :param map_file: coprus-key-map
    :param map_key: '$(id)
    :param default_output: 0.0
    :return:
    """
    # copy original net
    net = rasr.FlowNetwork(name=feature_net.name)
    net.add_param(["id", "start-time", "end-time"])
    mapping = net.add_net(feature_net)
    net.interconnect_inputs(feature_net, mapping)
    net.interconnect_outputs(feature_net, mapping)

    if map_key.startswith("$(") and map_key.endswith(")"):
        net.add_param(map_key[2:-1])

    net.add_output("labels")
    corpus_map = net.add_node(
        "generic-coprus-key-map",
        "warping-factor",
        {
            "key": map_key,
            "map-file": map_file,
            "default-output": "%s" % default_output,
            "start-time": "$(start-time)",
            "end-time": "$(end-time)",
        },
    )
    net.link(corpus_map, "network:labels")

    return net


class UbmLabelMapJob(Job):
    """
    Create a coprus-key-map where each entry is 0
    """

    def __init__(self, segment_list, alphas=None):
        """
        Example:
        segment_job = corpus_recipes.SegmentCorpus(self.crp[corpus].corpus_config.file, 1)
        map_job = ubm.UbmWarpingMapJob(segment_list=segment_job.single_segment_files[1])
        """
        if not alphas:
            alphas = [0]
        self.alphas = alphas
        self.segment_list = segment_list

        self.alphas_file = self.output_path("alphas.xml")
        self.warping_map = self.output_path("warping_map.xml")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        segments = []
        with open(tk.uncached_path(self.segment_list)) as f:
            segments = [l.strip() for l in f.readlines()]

        with open(self.warping_map.get_path(), "wt") as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write("<coprus-key-map>\n")
            for seg in sorted(segments):
                f.write('  <map-item key="%s" value="%s"/>\n' % (seg, 0))
            f.write("</coprus-key-map>")

        with open(self.alphas_file.get_path(), "wt") as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n<vector-string>\n')
            for factor in sorted(self.alphas):
                f.write("  %s\n" % factor)
            f.write("</vector-string>")


class TrainUniversalBackgroundModelSequence:
    """
    Similar to vtln.TrainWarpingFactorsSequence but we use ubm.EstimateUniversalBackgroundMixturesJob instead of vtln.EstimateWarpingMixturesJob
    """

    def __init__(
        self,
        crp,
        initial_mixtures,
        feature_flow,
        warping_map,
        warping_factors,
        action_sequence,
        split_extra_args=None,
        accumulate_extra_args=None,
        seq_extra_args=None,
    ):
        split_extra_args = {} if split_extra_args is None else split_extra_args
        accumulate_extra_args = (
            {} if accumulate_extra_args is None else accumulate_extra_args
        )
        seq_extra_args = {} if seq_extra_args is None else seq_extra_args

        self.action_sequence = action_sequence

        self.all_jobs = []
        self.all_logs = []
        self.all_mixtures = []
        self.selected_mixtures = []

        current_mixtures = initial_mixtures

        for idx, action in enumerate(action_sequence):
            split = action.startswith("split")
            args = {
                "crp": crp,
                "old_mixtures": current_mixtures,
                "feature_flow": feature_flow,
                "warping_map": warping_map,
                "warping_factors": warping_factors,
                "split_first": split,
            }
            args.update(split_extra_args if split else accumulate_extra_args)
            if idx in seq_extra_args:
                args.update(seq_extra_args[idx])

            j = EstimateUniversalBackgroundMixturesJob(**args)
            self.all_jobs.append(j)
            self.all_logs.append(j.log_file)
            self.all_mixtures.append(j.out_mixtures)

            current_mixtures = j.out_mixtures

            if action[-1] == "!":
                self.selected_mixtures.append(j.out_mixtures)


class EstimateUniversalBackgroundMixturesJob(mm.MergeMixturesJob):
    """
    Similar to vtln.EstimateWarpingMixturesJob but with the additional Rasr configuration:
      config.acoustic_model_trainer.mixture_set_trainer.covariance_tying = 'none'
      config.acoustic_model_trainer.mixture_set_trainer.force_covariance_tying = True
      config.acoustic_model_trainer.mixture_set_trainer.minimum_observation_weight = min_observation_weight
      config.acoustic_model_trainer.mixture_set_trainer.splitter.minimum_mean_observation_weight = min_observation_weight
      config.acoustic_model_trainer.mixture_set_trainer.splitter.minimum_covariance_observation_weight = min_observation_weight
    """

    def __init__(
        self,
        crp,
        old_mixtures,
        feature_flow,
        warping_map,
        warping_factors,
        min_observation_weight=20,
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

        (
            self.config,
            self.post_config,
        ) = EstimateUniversalBackgroundMixturesJob.create_config(**kwargs)
        self.label_flow = EstimateUniversalBackgroundMixturesJob.create_flow(**kwargs)
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
        min_observation_weight,
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

        config.acoustic_model_trainer.mixture_set_trainer.covariance_tying = "none"
        config.acoustic_model_trainer.mixture_set_trainer.force_covariance_tying = True
        config.acoustic_model_trainer.mixture_set_trainer.minimum_observation_weight = (
            min_observation_weight
        )
        config.acoustic_model_trainer.mixture_set_trainer.splitter.minimum_mean_observation_weight = (
            min_observation_weight
        )
        config.acoustic_model_trainer.mixture_set_trainer.splitter.minimum_covariance_observation_weight = (
            min_observation_weight
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
