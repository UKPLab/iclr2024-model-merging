__all__ = ["ScoreFeaturesJob"]

import math
import xml.etree.ElementTree as ET

from sisyphus import *

import i6_core.rasr as rasr
import i6_core.util as util

Path = setup_path(__package__)


class ScoreFeaturesJob(rasr.RasrCommand, Job):
    """

    This job uses a (multiple) RASR process(es) to forward all the data and calculates the average softmax output.
    It can be used to calculate the prior based on the "Povey" method.
    """

    def __init__(
        self,
        crp,
        feature_flow,
        feature_scorer,
        normalize=True,
        plot_prior=False,
        use_gpu=False,
        rtf=2.0,
        extra_config=None,
        extra_post_config=None,
        rqmt=None,
    ):
        """
        :param rasr.crp.CommonRasrParameters crp:
        :param rasr.flow.FlowNetwork feature_flow:
        :param rasr.feature_scorer.FeatureScorer feature_scorer:
        :param bool normalize:
        :param bool plot_prior:
        :param bool use_gpu:
        :param float rtf:
        :param rasr.config.RasrConfig|None extra_config:
        :param rasr.config.RasrConfig|None extra_post_config:
        :param dict|None rqmt:
        """
        assert isinstance(feature_scorer, rasr.FeatureScorer)

        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = ScoreFeaturesJob.create_config(**kwargs)
        self.feature_flow = feature_flow
        self.concurrent = crp.concurrent
        self.exe = self.select_exe(
            crp.acoustic_model_trainer_exe, "acoustic-model-trainer"
        )
        self.feature_scorer = feature_scorer
        self.normalize = normalize
        self.plot_prior = plot_prior
        self.use_gpu = use_gpu

        self.out_log_file = self.log_file_output_path("score_features", crp, True)
        self.out_prior = self.output_path("prior.xml", cached=True)
        if self.plot_prior:
            self.out_prior_plot = self.output_path("prior.png")

        self.rqmt = {
            "time": max(rtf * crp.corpus_duration / crp.concurrent, 0.5),
            "cpu": 1,
            "gpu": 1 if self.use_gpu else 0,
            "mem": 2,
        }
        if rqmt:
            self.rqmt.update(rqmt)

    def tasks(self):
        yield Task("create_files", resume="create_files", mini_task=True)
        yield Task(
            "run", resume="run", rqmt=self.rqmt, args=range(1, self.concurrent + 1)
        )
        yield Task("prior_and_plot", resume="prior_and_plot", mini_task=True)

    def create_files(self):
        self.write_config(self.config, self.post_config, "score_features.config")
        self.feature_flow.write_to_file("feature.flow")
        extra_code = (
            ":${{THEANO_FLAGS:="
            "}}\n"
            'export THEANO_FLAGS="$THEANO_FLAGS,device={0},force_device=True"\n'
            'export TF_DEVICE="{0}"'.format("gpu" if self.use_gpu else "cpu")
        )
        if self.rqmt["cpu"] > 1:
            extra_code += (
                "\nexport MKL_NUM_THREADS={}\nexport OMP_NUM_THREADS={}".format(
                    self.rqmt["cpu"], self.rqmt["cpu"]
                )
            )
        self.write_run_script(self.exe, "score_features.config", extra_code=extra_code)

    def run(self, task_id):
        self.run_script(task_id, self.out_log_file[task_id])

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("score_features.log.%d" % task_id)

    def prior_and_plot(self):
        all_scores = []
        all_frames = []
        max_emission = 0
        for l in self.out_log_file.values():
            with util.uopen(l.get_path(), "rb") as f:
                tree = ET.parse(f)
            scores = {}
            activations = tree.find(".//activations")
            all_frames.append(int(activations.find("num_frames").text.strip()))
            for score_elem in activations.findall("score"):
                emission = int(score_elem.attrib["emission"])
                max_emission = max(max_emission, emission)
                score = float(score_elem.text.strip())
                scores[emission] = score
            all_scores.append(scores)

        total_frames = sum(all_frames)
        merged_scores = [0.0 for _ in range(max_emission + 1)]
        for frames, scores in zip(all_frames, all_scores):
            scale = frames / total_frames
            for k, v in scores.items():
                merged_scores[k] += scale * v

        total_mass = sum(merged_scores)
        merged_scores = [s / total_mass for s in merged_scores]

        with open(self.out_prior.get_path(), "wt") as f:
            f.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="%d">\n'
                % len(merged_scores)
            )
            f.write(" ".join("%.20e" % math.log(s) for s in merged_scores) + "\n")
            f.write("</vector-f32>")

        if self.plot_prior:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xdata = range(len(merged_scores))
            plt.semilogy(xdata, merged_scores)

            plt.xlabel("emission idx")
            plt.ylabel("prior")
            plt.grid(True)
            plt.savefig(self.out_prior_plot.get_path())

    @classmethod
    def create_config(
        cls,
        crp,
        feature_flow,
        feature_scorer,
        extra_config,
        extra_post_config,
        **kwargs,
    ):
        """
        :param rasr.crp.CommonRasrParameters crp:
        :param rasr.flow.FlowNetwork feature_flow:
        :param rasr.feature_scorer.FeatureScorer feature_scorer:
        :param rasr.config.RasrConfig|None extra_config:
        :param rasr.config.RasrConfig|None extra_post_config:
        :param kwargs:
        :return:
        :rtype (rasr.config.RasrConfig, rasr.config.RasrConfig)
        """
        config, post_config = rasr.build_config_from_mapping(
            crp, {"corpus": "acoustic-model-trainer.corpus"}, parallelize=True
        )

        feature_scorer.apply_config(
            "acoustic-model-trainer.average-feature-scorer-activation.mixture-set",
            config,
            post_config,
        )
        feature_flow.apply_config(
            "acoustic-model-trainer.average-feature-scorer-activation.feature-extraction",
            config,
            post_config,
        )

        config.action = "calculate-average-feature-scorer-activation"
        config.acoustic_model_trainer.average_feature_scorer_activation.feature_extraction.file = (
            "feature.flow"
        )
        config.acoustic_model_trainer.average_feature_scorer_activation.output.channel = (
            crp.default_log_channel
        )

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        return super().hash(
            {
                "config": config,
                "feature_flow": kwargs["feature_flow"],
                "exe": kwargs["crp"].acoustic_model_trainer_exe,
                "normalize": kwargs["normalize"],
            }
        )
