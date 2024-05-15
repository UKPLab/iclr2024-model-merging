__all__ = [
    "FeatureScorer",
    "GMMFeatureScorer",
    "DiagonalMaximumScorer",
    "SimdDiagonalMaximumScorer",
    "PreselectionBatchIntScorer",
    "ReturnnScorer",
    "InvAlignmentPassThroughFeatureScorer",
    "PrecomputedHybridFeatureScorer",
]

from sisyphus import *

Path = setup_path(__package__)

import os

from .config import *


class FeatureScorer:
    def __init__(self):
        self.config = RasrConfig()
        self.post_config = RasrConfig()

    def apply_config(self, path, config, post_config):
        config[path]._update(self.config)
        post_config[path]._update(self.post_config)

    def html(self):
        config = repr(self.config).replace("\n", "<br />\n")
        post_config = repr(self.post_config).replace("\n", "<br />\n")
        return "<h3>Config:</h3>\n%s<br />\n<h3>Post Config:</h3>\n%s" % (
            config,
            post_config,
        )


class GMMFeatureScorer(FeatureScorer):
    def __init__(self, mixtures, scale=1.0):
        super().__init__()
        self.config.scale = scale
        self.config.file = mixtures


class DiagonalMaximumScorer(GMMFeatureScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.feature_scorer_type = "diagonal-maximum"


class SimdDiagonalMaximumScorer(GMMFeatureScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.feature_scorer_type = "SIMD-diagonal-maximum"


class PreselectionBatchIntScorer(GMMFeatureScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.feature_scorer_type = "preselection-batch-int"


class ReturnnScorer(FeatureScorer):
    def __init__(
        self,
        feature_dimension,
        output_dimension,
        prior_mixtures,
        model,
        mixture_scale=1.0,
        prior_scale=1.0,
        prior_file=None,
    ):
        super().__init__()

        self.config.feature_dimension = feature_dimension
        self.config.feature_scorer_type = "nn-trainer-feature-scorer"
        self.config.file = prior_mixtures
        self.config.priori_scale = prior_scale
        if prior_file is not None:
            self.config.prior_file = prior_file
        else:
            self.config.normalize_mixture_weights = False
        self.config.pymod_name = "returnn.SprintInterface"
        self.config.pymod_path = os.path.join(tk.gs.RETURNN_ROOT, "..")
        self.config.pymod_config = StringWrapper(
            "epoch:%d,action:forward,configfile:%s"
            % (model.epoch, model.returnn_config_file),
            model,
        )
        self.config.scale = mixture_scale
        self.config.target_mode = "forward-only"
        self.config.trainer = "python-trainer"
        self.config.trainer_output_dimension = output_dimension
        self.config.use_network = False

        self.returnn_config = model.returnn_config_file


class InvAlignmentPassThroughFeatureScorer(FeatureScorer):
    def __init__(self, prior_mixtures, max_segment_length, mapping, priori_scale=0.0):
        super().__init__()

        self.config = RasrConfig()
        self.config.feature_scorer_type = "inv-alignment-pass-through"
        self.config.file = prior_mixtures
        self.config.max_segment_length = max_segment_length
        self.config.mapping = mapping
        self.config.priori_scale = priori_scale
        self.config.normalize_mixture_weights = False


class PrecomputedHybridFeatureScorer(FeatureScorer):
    def __init__(self, prior_mixtures, scale=1.0, priori_scale=0.0, prior_file=None):
        super().__init__()

        self.config = RasrConfig()
        self.config.feature_scorer_type = "nn-precomputed-hybrid"
        self.config.file = prior_mixtures
        self.config.scale = scale
        if prior_file is not None:
            self.config.prior_file = prior_file
        self.config.priori_scale = priori_scale
        if prior_file is not None:
            self.config.prior_file = prior_file
        self.config.normalize_mixture_weights = False


class TFLabelContextFeatureScorer(FeatureScorer):
    def __init__(
        self,
        fs_tf_config,
        contextPriorFile,
        diphonePriorFile,
        prior_mixtures,
        prior_scale,
    ):
        super().__init__()

        self.config = RasrConfig()
        self.config.feature_scorer_type = "tf-label-context-scorer"
        self.config.file = prior_mixtures
        self.config.num_label_contexts = 46
        self.config.prior_scale = prior_scale
        self.config.context_prior = contextPriorFile
        self.config.diphone_prior = diphonePriorFile
        self.config.normalize_mixture_weights = False
        self.config.loader = fs_tf_config.loader
        self.config.input_map = fs_tf_config.input_map
        self.config.output_map = fs_tf_config.output_map
