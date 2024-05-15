__all__ = [
    "AdvancedTreeSearchLmImageAndGlobalCacheJob",
    "AdvancedTreeSearchJob",
    "AdvancedTreeSearchWithRescoringJob",
    "BidirectionalAdvancedTreeSearchJob",
    "BuildGlobalCacheJob",
]

from sisyphus import *

Path = setup_path(__package__)

import math
import os
import shutil

import i6_core.lm as lm
import i6_core.rasr as rasr
import i6_core.util as util


class AdvancedTreeSearchLmImageAndGlobalCacheJob(rasr.RasrCommand, Job):
    def __init__(self, crp, feature_scorer, extra_config=None, extra_post_config=None):
        assert isinstance(feature_scorer, rasr.FeatureScorer)

        self.set_vis_name("Precompute LM Image/Global Cache")

        kwargs = locals()
        del kwargs["self"]

        (
            self.config,
            self.post_config,
            self.num_images,
        ) = AdvancedTreeSearchLmImageAndGlobalCacheJob.create_config(**kwargs)
        self.exe = self.select_exe(crp.flf_tool_exe, "flf-tool")

        self.out_log_file = self.log_file_output_path("lm_and_state_tree", crp, False)
        self.out_lm_images = {
            i: self.output_path("lm-%d.image" % i, cached=True)
            for i in range(1, self.num_images + 1)
        }
        self.out_global_cache = self.output_path("global.cache", cached=True)

        self.rqmt = {"time": 1, "cpu": 1, "mem": 2}

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def create_files(self):
        self.write_config(self.config, self.post_config, "lm_and_global_cache.config")
        with open("dummy.corpus", "wt") as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8" ?>\n<corpus name="dummy"></corpus>'
            )
        with open("dummy.flow", "wt") as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8" ?>\n<network><out name="features" /></network>'
            )
        extra_code = (
            ":${THEANO_FLAGS:="
            '}\nexport THEANO_FLAGS="$THEANO_FLAGS,device=cpu,force_device=True"\nexport TF_DEVICE="cpu"'
        )
        self.write_run_script(
            self.exe, "lm_and_global_cache.config", extra_code=extra_code
        )

    def run(self):
        self.run_script(1, self.out_log_file)
        for i in range(1, self.num_images + 1):
            shutil.move("lm-%d.image" % i, self.out_lm_images[i].get_path())
        shutil.move("global.cache", self.out_global_cache.get_path())

    def cleanup_before_run(self, cmd, retry, *args):
        util.backup_if_exists("lm_and_state_tree.log")

    @classmethod
    def find_arpa_lms(cls, lm_config, lm_post_config=None):
        result = []

        def has_image(c, pc):
            res = c._get("image") is not None
            res = res or (pc is not None and pc._get("image") is not None)
            return res

        if lm_config.type == "ARPA":
            if not has_image(lm_config, lm_post_config):
                result.append((lm_config, lm_post_config))
        elif lm_config.type == "combine":
            for i in range(1, lm_config.num_lms + 1):
                sub_lm_config = lm_config["lm-%d" % i]
                sub_lm_post_config = (
                    lm_post_config["lm-%d" % i] if lm_post_config is not None else None
                )
                result += cls.find_arpa_lms(sub_lm_config, sub_lm_post_config)
        return result

    @classmethod
    def create_config(
        cls, crp, feature_scorer, extra_config, extra_post_config, **kwargs
    ):
        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "lexicon": "flf-lattice-tool.lexicon",
                "acoustic_model": "flf-lattice-tool.network.recognizer.acoustic-model",
                "language_model": "flf-lattice-tool.network.recognizer.lm",
            },
        )

        # the length model does not matter for the global.cache, remove it
        del config.flf_lattice_tool.network.recognizer.acoustic_model["length"]

        config.flf_lattice_tool.network.initial_nodes = "segment"

        config.flf_lattice_tool.network.segment.type = "speech-segment"
        config.flf_lattice_tool.network.segment.links = "1->recognizer:1"

        config.flf_lattice_tool.corpus.file = "dummy.corpus"
        config.flf_lattice_tool.network.recognizer.type = "recognizer"
        config.flf_lattice_tool.network.recognizer.links = "sink"
        config.flf_lattice_tool.network.recognizer.apply_non_word_closure_filter = False
        config.flf_lattice_tool.network.recognizer.add_confidence_score = False
        config.flf_lattice_tool.network.recognizer.apply_posterior_pruning = False
        config.flf_lattice_tool.network.recognizer.search_type = "advanced-tree-search"
        config.flf_lattice_tool.network.recognizer.feature_extraction.file = (
            "dummy.flow"
        )
        config.flf_lattice_tool.network.recognizer.lm.scale = 1.0

        arpa_lms = cls.find_arpa_lms(
            config.flf_lattice_tool.network.recognizer.lm,
            post_config.flf_lattice_tool.network.recognizer.lm
            if post_config is not None
            else None,
        )
        for i, lm_config in enumerate(arpa_lms):
            lm_config[0].image = "lm-%d.image" % (i + 1)

        config.flf_lattice_tool.global_cache.file = "global.cache"

        # in this job the feature scorer does not matter, but is required by rasr
        feature_scorer.apply_config(
            "flf-lattice-tool.network.recognizer.acoustic-model.mixture-set",
            post_config,
            post_config,
        )

        config.flf_lattice_tool.network.sink.type = "sink"
        post_config.flf_lattice_tool.network.sink.warn_on_empty_lattice = True
        post_config.flf_lattice_tool.network.sink.error_on_empty_lattice = False

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config, len(arpa_lms)

    @classmethod
    def hash(cls, kwargs):
        config, post_config, num_images = cls.create_config(**kwargs)
        return super().hash({"config": config, "exe": kwargs["crp"].flf_tool_exe})


class AdvancedTreeSearchJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        feature_flow,
        feature_scorer,
        search_parameters=None,
        lm_lookahead=True,
        lookahead_options=None,
        create_lattice=True,
        eval_single_best=True,
        eval_best_in_lattice=True,
        use_gpu=False,
        rtf=10,
        mem=4,
        cpu=1,
        lmgc_mem=12,
        model_combination_config=None,
        model_combination_post_config=None,
        extra_config=None,
        extra_post_config=None,
    ):
        assert isinstance(feature_scorer, rasr.FeatureScorer)

        self.set_vis_name("Advanced Beam Search")

        kwargs = locals()
        del kwargs["self"]

        (
            self.config,
            self.post_config,
            self.lm_gc_job,
        ) = AdvancedTreeSearchJob.create_config(**kwargs)
        self.feature_flow = feature_flow
        self.exe = self.select_exe(crp.flf_tool_exe, "flf-tool")
        self.concurrent = crp.concurrent
        self.use_gpu = use_gpu

        self.out_log_file = self.log_file_output_path("search", crp, True)
        self.out_single_lattice_caches = dict(
            (task_id, self.output_path("lattice.cache.%d" % task_id, cached=True))
            for task_id in range(1, crp.concurrent + 1)
        )
        self.out_lattice_bundle = self.output_path("lattice.bundle", cached=True)
        self.out_lattice_path = util.MultiOutputPath(
            self, "lattice.cache.$(TASK)", self.out_single_lattice_caches, cached=True
        )
        self.cpu = cpu
        self.rqmt = {
            "time": max(crp.corpus_duration * rtf / crp.concurrent, 0.5),
            "cpu": cpu,
            "gpu": 1 if self.use_gpu else 0,
            "mem": mem,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task(
            "run", resume="run", rqmt=self.rqmt, args=range(1, self.concurrent + 1)
        )

    def create_files(self):
        self.write_config(self.config, self.post_config, "recognition.config")
        self.feature_flow.write_to_file("feature.flow")
        util.write_paths_to_file(
            self.out_lattice_bundle, self.out_single_lattice_caches.values()
        )
        extra_code = "export OMP_NUM_THREADS={0}\nexport TF_DEVICE='{1}'".format(
            math.ceil(self.cpu / 2), "gpu" if self.use_gpu else "cpu"
        )
        self.write_run_script(self.exe, "recognition.config", extra_code=extra_code)

    def run(self, task_id):
        self.run_script(task_id, self.out_log_file[task_id])
        shutil.move(
            "lattice.cache.%d" % task_id,
            self.out_single_lattice_caches[task_id].get_path(),
        )

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("search.log.%d" % task_id)
        util.delete_if_exists("lattice.cache.%d" % task_id)

    @classmethod
    def update_search_parameters(cls, search_parameters):
        if search_parameters is None:
            search_parameters = {}

        default_search_parameters = {
            "beam-pruning": 15,
            "beam-pruning-limit": 100000,
            "word-end-pruning": 0.5,
            "word-end-pruning-limit": 10000,
        }
        default_search_parameters.update(search_parameters)

        return default_search_parameters

    @classmethod
    def create_config(
        cls,
        crp,
        feature_flow,
        feature_scorer,
        search_parameters,
        lm_lookahead,
        lookahead_options,
        create_lattice,
        eval_single_best,
        eval_best_in_lattice,
        mem,
        cpu,
        lmgc_mem,
        model_combination_config,
        model_combination_post_config,
        extra_config,
        extra_post_config,
        **kwargs,
    ):
        lm_gc = AdvancedTreeSearchLmImageAndGlobalCacheJob(
            crp, feature_scorer, extra_config, extra_post_config
        )
        lm_gc.rqmt["mem"] = lmgc_mem

        search_parameters = cls.update_search_parameters(search_parameters)

        la_opts = {
            "history_limit": 1,
            "tree_cutoff": 30,
            "minimum_representation": 1,
            "cache_low": 2000,
            "cache_high": 3000,
            "laziness": 15,
        }
        if lookahead_options is not None:
            la_opts.update(lookahead_options)

        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "corpus": "flf-lattice-tool.corpus",
                "lexicon": "flf-lattice-tool.lexicon",
                "acoustic_model": "flf-lattice-tool.network.recognizer.acoustic-model",
                "language_model": "flf-lattice-tool.network.recognizer.lm",
            },
            parallelize=True,
        )

        config.flf_lattice_tool.network.initial_nodes = "segment"
        config.flf_lattice_tool.network.segment.type = "speech-segment"
        config.flf_lattice_tool.network.segment.links = (
            "1->recognizer:1 0->archive-writer:1"
            + (" 0->evaluator:1" if eval_single_best or eval_best_in_lattice else "")
        )

        config.flf_lattice_tool.network.recognizer.type = "recognizer"
        config.flf_lattice_tool.network.recognizer.links = "expand"

        # Parameters for Flf::Recognizer
        config.flf_lattice_tool.network.recognizer.apply_non_word_closure_filter = False
        config.flf_lattice_tool.network.recognizer.add_confidence_score = False
        config.flf_lattice_tool.network.recognizer.apply_posterior_pruning = False

        # Parameters for Speech::Recognizer
        config.flf_lattice_tool.network.recognizer.search_type = "advanced-tree-search"

        # Parameters for Speech::DataSource or Sparse::DataSource
        config.flf_lattice_tool.network.recognizer.feature_extraction.file = (
            "feature.flow"
        )
        feature_flow.apply_config(
            "flf-lattice-tool.network.recognizer.feature-extraction",
            config,
            post_config,
        )

        # Parameters for Am::ClassicAcousticModel
        feature_scorer.apply_config(
            "flf-lattice-tool.network.recognizer.acoustic-model.mixture-set",
            config,
            post_config,
        )

        # Parameters for Speech::Model combination (besides AM and LM parameters)
        config.flf_lattice_tool.network.recognizer._update(model_combination_config)
        post_config.flf_lattice_tool.network.recognizer._update(
            model_combination_post_config
        )

        # Search parameters
        config.flf_lattice_tool.network.recognizer.recognizer.create_lattice = (
            create_lattice
        )

        config.flf_lattice_tool.network.recognizer.recognizer.beam_pruning = (
            search_parameters["beam-pruning"]
        )
        config.flf_lattice_tool.network.recognizer.recognizer.beam_pruning_limit = (
            search_parameters["beam-pruning-limit"]
        )
        config.flf_lattice_tool.network.recognizer.recognizer.word_end_pruning = (
            search_parameters["word-end-pruning"]
        )
        config.flf_lattice_tool.network.recognizer.recognizer.word_end_pruning_limit = (
            search_parameters["word-end-pruning-limit"]
        )
        if "lm-state-pruning" in search_parameters:
            config.flf_lattice_tool.network.recognizer.recognizer.lm_state_pruning = (
                search_parameters["lm-state-pruning"]
            )

        config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead = (
            rasr.RasrConfig()
        )
        config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead._value = (
            lm_lookahead
        )
        config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead_laziness = (
            la_opts["laziness"]
        )
        config.flf_lattice_tool.network.recognizer.recognizer.optimize_lattice = (
            "simple"
        )
        if lm_lookahead:
            config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead.history_limit = la_opts[
                "history_limit"
            ]
            config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead.tree_cutoff = la_opts[
                "tree_cutoff"
            ]
            config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead.minimum_representation = la_opts[
                "minimum_representation"
            ]
            if "lm_lookahead_scale" in la_opts:
                config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead.lm_lookahead_scale = la_opts[
                    "lm_lookahead_scale"
                ]
            post_config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead.cache_size_low = la_opts[
                "cache_low"
            ]
            post_config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead.cache_size_high = la_opts[
                "cache_high"
            ]

        post_config.flf_lattice_tool.global_cache.read_only = True
        post_config.flf_lattice_tool.global_cache.file = lm_gc.out_global_cache

        arpa_lms = AdvancedTreeSearchLmImageAndGlobalCacheJob.find_arpa_lms(
            config.flf_lattice_tool.network.recognizer.lm,
            post_config.flf_lattice_tool.network.recognizer.lm,
        )
        for i, lm_config in enumerate(arpa_lms):
            lm_config[1].image = lm_gc.out_lm_images[i + 1]

        # Remaining Flf-network

        config.flf_lattice_tool.network.expand.type = "expand-transits"
        if eval_single_best or eval_best_in_lattice:
            config.flf_lattice_tool.network.expand.links = "evaluator archive-writer"

            config.flf_lattice_tool.network.evaluator.type = "evaluator"
            config.flf_lattice_tool.network.evaluator.links = "sink:0"
            config.flf_lattice_tool.network.evaluator.word_errors = True
            config.flf_lattice_tool.network.evaluator.single_best = eval_single_best
            config.flf_lattice_tool.network.evaluator.best_in_lattice = (
                eval_best_in_lattice
            )
            config.flf_lattice_tool.network.evaluator.edit_distance.format = "bliss"
            config.flf_lattice_tool.network.evaluator.edit_distance.allow_broken_words = (
                False
            )
        else:
            config.flf_lattice_tool.network.expand.links = "archive-writer"

        config.flf_lattice_tool.network.archive_writer.type = "archive-writer"
        config.flf_lattice_tool.network.archive_writer.links = "sink:1"
        config.flf_lattice_tool.network.archive_writer.format = "flf"
        config.flf_lattice_tool.network.archive_writer.path = "lattice.cache.$(TASK)"
        post_config.flf_lattice_tool.network.archive_writer.info = True

        config.flf_lattice_tool.network.sink.type = "sink"
        post_config.flf_lattice_tool.network.sink.warn_on_empty_lattice = True
        post_config.flf_lattice_tool.network.sink.error_on_empty_lattice = False

        post_config["*"].session.intra_op_parallelism_threads = cpu
        post_config["*"].session.inter_op_parallelism_threads = 1

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config, lm_gc

    @classmethod
    def hash(cls, kwargs):
        config, post_config, lm_gc = cls.create_config(**kwargs)
        return super().hash(
            {
                "config": config,
                "feature_flow": kwargs["feature_flow"],
                "exe": kwargs["crp"].flf_tool_exe,
            }
        )


class AdvancedTreeSearchWithRescoringJob(AdvancedTreeSearchJob):
    def __init__(
        self,
        crp,
        feature_flow,
        feature_scorer,
        search_parameters=None,
        lm_lookahead=True,
        lookahead_options=None,
        create_lattice=True,
        eval_single_best=True,
        eval_best_in_lattice=True,
        use_gpu=False,
        rtf=10,
        mem=4,
        model_combination_config=None,
        model_combination_post_config=None,
        rescorer_type="single-best",
        rescoring_lm_config=None,
        max_hypotheses=5,
        pruning_threshold=16.0,
        history_limit=0,
        rescoring_lookahead_scale=1.0,
        extra_config=None,
        extra_post_config=None,
    ):
        super().__init__(
            crp=crp,
            feature_flow=feature_flow,
            feature_scorer=feature_scorer,
            search_parameters=search_parameters,
            lm_lookahead=lm_lookahead,
            lookahead_options=lookahead_options,
            create_lattice=create_lattice,
            eval_single_best=eval_single_best,
            eval_best_in_lattice=eval_best_in_lattice,
            use_gpu=use_gpu,
            rtf=rtf,
            mem=mem,
            model_combination_config=model_combination_config,
            model_combination_post_config=model_combination_post_config,
            extra_config=extra_config,
            extra_post_config=extra_post_config,
        )
        self.set_vis_name("Advanced Beam Search with Rescoring")

        kwargs = locals()
        del kwargs["self"]

        (
            self.config,
            self.post_config,
            self.lm_gc_job,
        ) = AdvancedTreeSearchWithRescoringJob.create_config(**kwargs)

    @classmethod
    def create_config(
        cls,
        rescorer_type,
        rescoring_lm_config,
        max_hypotheses,
        pruning_threshold,
        history_limit,
        rescoring_lookahead_scale,
        **kwargs,
    ):
        config, post_config, lm_gc_job = super().create_config(**kwargs)

        config.flf_lattice_tool.network.recognizer.links = "rescore"

        rescore_config = config.flf_lattice_tool.network.rescore
        rescore_config.type = "push-forward-rescoring"
        rescore_config.links = "expand"
        rescore_config.key = "lm"
        rescore_config.rescorer_type = rescorer_type
        rescore_config.max_hypotheses = max_hypotheses
        rescore_config.pruning_threshold = pruning_threshold
        rescore_config.history_limit = history_limit
        rescore_config.lookahead_scale = rescoring_lookahead_scale
        rescore_config.lm = rescoring_lm_config

        return config, post_config, lm_gc_job


class BidirectionalAdvancedTreeSearchJob(rasr.RasrCommand, Job):
    # TODO: add proper create cache and lm-image job for this

    def __init__(
        self,
        crp,
        feature_flow,
        feature_scorer,
        recognizer_parameters=None,
        search_parameters=None,
        lm_lookahead=True,
        lookahead_options=None,
        create_lattice=True,
        lattice_filter_type="dummy",
        eval_single_best=True,
        eval_best_in_lattice=True,
        use_gpu=False,
        rtf=10,
        mem=4,
        model_combination_config=None,
        model_combination_post_config=None,
        extra_config=None,
        extra_post_config=None,
    ):
        assert isinstance(feature_scorer, rasr.FeatureScorer)

        self.set_vis_name("Bidirectional Advanced Beam Search")

        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = self.create_config(**kwargs)
        self.feature_flow = feature_flow
        self.exe = self.select_exe(crp.flf_tool_exe, "flf-tool")
        self.concurrent = crp.concurrent
        self.use_gpu = use_gpu

        self.out_log_file = self.log_file_output_path("search", crp, True)
        self.out_single_lattice_caches = dict(
            (task_id, self.output_path("lattice.cache.%d" % task_id, cached=True))
            for task_id in range(1, crp.concurrent + 1)
        )
        self.out_lattice_bundle = self.output_path("lattice.bundle", cached=True)
        self.out_lattice_path = util.MultiOutputPath(
            self, "lattice.cache.$(TASK)", self.out_single_lattice_caches, cached=True
        )

        self.rqmt = {
            "time": max(crp.corpus_duration * rtf / crp.concurrent, 0.5),
            "cpu": 1,
            "gpu": 1 if self.use_gpu else 0,
            "mem": mem,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task(
            "run", resume="run", rqmt=self.rqmt, args=range(1, self.concurrent + 1)
        )

    def create_files(self):
        self.write_config(self.config, self.post_config, "recognition.config")
        self.feature_flow.write_to_file("feature.flow")
        util.write_paths_to_file(
            self.out_lattice_bundle, self.out_single_lattice_caches.values()
        )
        extra_code = (
            ":${{THEANO_FLAGS:="
            "}}\n"
            'export THEANO_FLAGS="$THEANO_FLAGS,device={0},force_device=True"\n'
            'export TF_DEVICE="{0}"'.format("gpu" if self.use_gpu else "cpu")
        )
        self.write_run_script(self.exe, "recognition.config", extra_code=extra_code)

    def run(self, task_id):
        self.run_script(task_id, self.out_log_file[task_id])
        shutil.move(
            "lattice.cache.%d" % task_id,
            self.out_single_lattice_caches[task_id].get_path(),
        )

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("search.log.%d" % task_id)
        util.delete_if_exists("lattice.cache.%d" % task_id)

    @classmethod
    def update_search_parameters(cls, search_parameters):
        if search_parameters is None:
            search_parameters = {}

        default_search_parameters = {
            "beam-pruning": 15,
            "beam-pruning-limit": 100000,
            "word-end-pruning": 0.5,
            "word-end-pruning-limit": 10000,
        }
        default_search_parameters.update(search_parameters)

        return default_search_parameters

    @classmethod
    def create_config(
        cls,
        crp,
        feature_flow,
        feature_scorer,
        recognizer_parameters,
        search_parameters,
        lm_lookahead,
        lookahead_options,
        create_lattice,
        lattice_filter_type,
        eval_single_best,
        eval_best_in_lattice,
        mem,
        model_combination_config,
        model_combination_post_config,
        extra_config,
        extra_post_config,
        **kwargs,
    ):
        lm_gc = AdvancedTreeSearchLmImageAndGlobalCacheJob(
            crp, feature_scorer, extra_config, extra_post_config
        )
        lm_gc.rqmt["mem"] = mem

        search_parameters = cls.update_search_parameters(search_parameters)

        la_opts = {
            "history_limit": 1,
            "tree_cutoff": 30,
            "minimum_representation": 1,
            "cache_low": 2000,
            "cache_high": 3000,
            "laziness": 15,
        }
        if lookahead_options is not None:
            la_opts.update(lookahead_options)

        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "corpus": "flf-lattice-tool.corpus",
                "lexicon": "flf-lattice-tool.lexicon",
                "acoustic_model": "flf-lattice-tool.network.recognizer.acoustic-model",
                "language_model": "flf-lattice-tool.network.recognizer.lm",
            },
            parallelize=True,
        )

        config.flf_lattice_tool.network.initial_nodes = "segment"
        config.flf_lattice_tool.network.segment.type = "speech-segment"
        config.flf_lattice_tool.network.segment.links = (
            "1->recognizer:1 0->archive-writer:1 0->evaluator:1"
        )

        config.flf_lattice_tool.network.recognizer.type = "incremental-recognizer"
        config.flf_lattice_tool.network.recognizer.links = "filter"

        # Parameters for Flf::Recognizer (incremental Recognizer?)
        config.flf_lattice_tool.network.recognizer.apply_non_word_closure_filter = False
        config.flf_lattice_tool.network.recognizer.add_confidence_score = False
        config.flf_lattice_tool.network.recognizer.apply_posterior_pruning = False

        # Parameters for Speech::Recognizer (incremental Recognizer?)
        config.flf_lattice_tool.network.recognizer.search_type = "advanced-tree-search"
        config.flf_lattice_tool.network.recognizer.backward.search_type = (
            "advanced-tree-search"
        )

        # Parameters for Speech::DataSource or Sparse::DataSource
        config.flf_lattice_tool.network.recognizer.feature_extraction.file = (
            "feature.flow"
        )
        feature_flow.apply_config(
            "flf-lattice-tool.network.recognizer.feature-extraction",
            config,
            post_config,
        )

        config.flf_lattice_tool.network.recognizer.backward.feature_extraction.file = (
            "feature.flow"
        )
        feature_flow.apply_config(
            "flf-lattice-tool.network.recognizer.backward.feature-extraction",
            config,
            post_config,
        )

        # Parameters for Am::ClassicAcousticModel
        feature_scorer.apply_config(
            "flf-lattice-tool.network.recognizer.acoustic-model.mixture-set",
            config,
            post_config,
        )

        # Parameters for Speech::Model combination (besides AM and LM parameters)
        config.flf_lattice_tool.network.recognizer._update(model_combination_config)
        post_config.flf_lattice_tool.network.recognizer._update(
            model_combination_post_config
        )
        config.flf_lattice_tool.network.recognizer.backward._update(
            model_combination_config
        )
        post_config.flf_lattice_tool.network.recognizer.backward._update(
            model_combination_post_config
        )

        # Search parameters
        config.flf_lattice_tool.network.recognizer.recognizer.create_lattice = (
            create_lattice
        )

        for recog_dir in ["recognizer", "backward"]:
            if recog_dir == "recognizer":
                recognizer_config = (
                    config.flf_lattice_tool.network.recognizer.recognizer
                )
                recognizer_post_config = (
                    post_config.flf_lattice_tool.network.recognizer.recognizer
                )
            else:
                recognizer_config = (
                    config.flf_lattice_tool.network.recognizer.backward.recognizer
                )
                recognizer_post_config = (
                    post_config.flf_lattice_tool.network.recognizer.backward.recognizer
                )

            recognizer_config.beam_pruning = search_parameters["beam-pruning"]
            recognizer_config.beam_pruning_limit = search_parameters[
                "beam-pruning-limit"
            ]
            recognizer_config.word_end_pruning = search_parameters["word-end-pruning"]
            recognizer_config.word_end_pruning_limit = search_parameters[
                "word-end-pruning-limit"
            ]

            recognizer_config.lm_lookahead = rasr.RasrConfig()
            recognizer_config.lm_lookahead._value = lm_lookahead
            recognizer_config.lm_lookahead_laziness = la_opts["laziness"]
            recognizer_config.optimize_lattice = "simple"
            if lm_lookahead:
                recognizer_config.lm_lookahead.history_limit = la_opts["history_limit"]
                recognizer_config.lm_lookahead.tree_cutoff = la_opts["tree_cutoff"]
                recognizer_config.lm_lookahead.minimum_representation = la_opts[
                    "minimum_representation"
                ]
                recognizer_post_config.lm_lookahead.cache_size_low = la_opts[
                    "cache_low"
                ]
                recognizer_post_config.lm_lookahead.cache_size_high = la_opts[
                    "cache_high"
                ]
            if recognizer_parameters is not None:
                for k in [
                    "decoder-initial-update-rate",
                    "lattice-relax-pruning-factor",
                    "lattice-relax-pruning-offset",
                    "correct-strict-initial",
                ]:
                    if k in recognizer_parameters:
                        recognizer_config[k] = self.recognizer_parameters[k]

        post_config.flf_lattice_tool.global_cache.read_only = True
        post_config.flf_lattice_tool.global_cache.file = lm_gc.out_global_cache
        post_config.flf_lattice_tool.network.recognizer.lm.image = lm_gc.lm_image
        post_config.flf_lattice_tool.network.recognizer.backward.lm.type = "ARPA"
        post_config.flf_lattice_tool.network.recognizer.backward.lm.file = (
            lm.ReverseARPALmJob(crp.language_model_config.file).reverse_lm_path
        )
        post_config.flf_lattice_tool.network.recognizer.backward.lm.scale = (
            crp.language_model_config.scale
        )

        # Remaining Flf-network

        config.flf_lattice_tool.network.filter.type = lattice_filter_type
        config.flf_lattice_tool.network.filter.links = "expand"

        config.flf_lattice_tool.network.expand.type = "expand-transits"
        config.flf_lattice_tool.network.expand.links = "evaluator archive-writer"

        config.flf_lattice_tool.network.evaluator.type = "evaluator"
        config.flf_lattice_tool.network.evaluator.links = "sink:0"
        config.flf_lattice_tool.network.evaluator.word_errors = True
        config.flf_lattice_tool.network.evaluator.single_best = eval_single_best
        config.flf_lattice_tool.network.evaluator.best_in_lattice = eval_best_in_lattice
        config.flf_lattice_tool.network.evaluator.edit_distance.format = "bliss"
        config.flf_lattice_tool.network.evaluator.edit_distance.allow_broken_words = (
            False
        )

        config.flf_lattice_tool.network.archive_writer.type = "archive-writer"
        config.flf_lattice_tool.network.archive_writer.links = "sink:1"
        config.flf_lattice_tool.network.archive_writer.format = "flf"
        config.flf_lattice_tool.network.archive_writer.path = "lattice.cache.$(TASK)"
        post_config.flf_lattice_tool.network.archive_writer.info = True

        config.flf_lattice_tool.network.sink.type = "sink"
        post_config.flf_lattice_tool.network.sink.warn_on_empty_lattice = True
        post_config.flf_lattice_tool.network.sink.error_on_empty_lattice = False

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
                "exe": kwargs["crp"].flf_tool_exe,
            }
        )


class BuildGlobalCacheJob(rasr.RasrCommand, Job):
    """
    Standalone job to create the global-cache for advanced-tree-search
    """

    def __init__(self, crp, extra_config=None, extra_post_config=None):
        """
        :param rasr.CommonRasrParameters crp: common RASR params (required: lexicon, acoustic_model, language_model, recognizer)
        :param rasr.Configuration extra_config: overlay config that influences the Job's hash
        :param rasr.Configuration extra_post_config: overlay config that does not influences the Job's hash
        """
        self.set_vis_name("Build Global Cache")

        kwargs = locals()
        del kwargs["self"]

        (
            self.config,
            self.post_config,
        ) = BuildGlobalCacheJob.create_config(**kwargs)
        self.exe = self.select_exe(crp.speech_recognizer_exe, "speech-recognizer")

        self.out_log_file = self.log_file_output_path("build_global_cache", crp, False)
        self.out_global_cache = self.output_path("global.cache", cached=True)

        self.rqmt = {"time": 1, "cpu": 1, "mem": 2}

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def create_files(self):
        self.write_config(self.config, self.post_config, "build_global_cache.config")
        self.write_run_script(self.exe, "build_global_cache.config")

    def run(self):
        self.run_script(1, self.out_log_file)
        shutil.move("global.cache", self.out_global_cache.get_path())

    def cleanup_before_run(self, cmd, retry, *args):
        util.backup_if_exists("build_global_cache.log")

    @classmethod
    def create_config(cls, crp, extra_config, extra_post_config, **kwargs):
        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "lexicon": "speech-recognizer.model-combination.lexicon",
                "acoustic_model": "speech-recognizer.model-combination.acoustic-model",
                "language_model": "speech-recognizer.model-combination.lm",
                "recognizer": "speech-recognizer.recognizer",
            },
        )

        config.speech_recognizer.recognition_mode = "init-only"
        config.speech_recognizer.search_type = "advanced-tree-search"
        config.speech_recognizer.global_cache.file = "global.cache"
        config.speech_recognizer.global_cache.read_only = False

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        return super().hash(
            {"config": config, "exe": kwargs["crp"].speech_recognizer_exe}
        )
