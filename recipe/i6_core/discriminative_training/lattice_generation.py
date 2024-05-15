__all__ = [
    "NumeratorLatticeJob",
    "RawDenominatorLatticeJob",
    "DenominatorLatticeJob",
    "StateAccuracyJob",
    "PhoneAccuracyJob",
]

import shutil
import os
import math

from sisyphus import *

Path = setup_path(__package__)

from i6_core.mm.flow import alignment_flow
import i6_core.mm as mm
import i6_core.recognition as recognition
import i6_core.util as util
import i6_core.rasr as rasr


class NumeratorLatticeJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        feature_flow,
        feature_scorer,
        alignment_options=None,
        use_gpu=False,
        rtf=10.0,
        extra_config=None,
        extra_post_config=None,
    ):

        assert isinstance(feature_scorer, rasr.FeatureScorer)

        self.set_vis_name("NumeratorLattice")

        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = NumeratorLatticeJob.create_config(**kwargs)
        self.alignment_flow = NumeratorLatticeJob.create_flow(**kwargs)
        self.exe = self.select_exe(
            crp.acoustic_model_trainer_exe, "acoustic-model-trainer"
        )
        self.concurrent = crp.concurrent
        self.feature_scorer = feature_scorer
        self.use_gpu = use_gpu

        self.log_file = self.log_file_output_path("create-numerator", crp, True)
        self.single_lattice_caches = {
            i: self.output_path("numerator.%d" % i, cached=True)
            for i in range(1, self.concurrent + 1)
        }
        self.lattice_path = util.MultiOutputPath(
            self, "numerator.$(TASK)", self.single_lattice_caches, cached=True
        )
        self.lattice_bundle = self.output_path("numerator.bundle", cached=True)

        self.rqmt = {
            "time": max(rtf * crp.corpus_duration / crp.concurrent, 0.5),
            "cpu": 2,
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
            self.lattice_bundle, self.single_lattice_caches.values()
        )
        extra_code = (
            ":${{THEANO_FLAGS:="
            "}}\n"
            'export THEANO_FLAGS="$THEANO_FLAGS,device={0},force_device=True"\n'
            'export TF_DEVICE="{0}"'.format("gpu" if self.use_gpu else "cpu")
        )
        self.write_run_script(self.exe, "alignment.config", extra_code=extra_code)

    def run(self, task_id):
        self.run_script(task_id, self.log_file[task_id])
        shutil.move(
            "numerator.%d" % task_id, self.single_lattice_caches[task_id].get_path()
        )

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("create-numerator.log.%d" % task_id)
        util.delete_if_zero("numerator.%d" % task_id)

    @classmethod
    def create_config(
        cls,
        crp,
        feature_flow,
        feature_scorer,
        alignment_options,
        extra_config,
        extra_post_config,
        **kwargs,
    ):
        alignment_flow = cls.create_flow(feature_flow)

        alignopt = {
            "increase-pruning-until-no-score-difference": True,
            "min-acoustic-pruning": 500,
            "max-acoustic-pruning": 10000,
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

            node_config.store_lattices = True
            node_config.lattice_archive.path = "numerator.$(TASK)"

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
        return alignment_flow(feature_flow, None)

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


class RawDenominatorLatticeJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        feature_flow,
        feature_scorer,
        search_parameters=None,
        lm_lookahead=True,
        lookahead_options=None,
        use_gpu=False,
        rtf=30,
        mem=4,
        model_combination_config=None,
        model_combination_post_config=None,
        extra_config=None,
        extra_post_config=None,
    ):

        assert isinstance(feature_scorer, rasr.FeatureScorer)

        self.set_vis_name("Raw Denominator Lattice")

        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = self.create_config(**kwargs)
        self.feature_flow = feature_flow
        self.exe = self.select_exe(crp.speech_recognizer_exe, "speech-recognizer")
        self.concurrent = crp.concurrent
        self.use_gpu = use_gpu

        self.log_file = self.log_file_output_path("create-raw-denominator", crp, True)
        self.single_lattice_caches = {
            task_id: self.output_path("raw-denominator.%d" % task_id, cached=True)
            for task_id in range(1, crp.concurrent + 1)
        }
        self.lattice_bundle = self.output_path("raw-denominator.bundle", cached=True)
        self.lattice_path = util.MultiOutputPath(
            self, "raw-denominator.$(TASK)", self.single_lattice_caches, cached=True
        )

        self.rqmt = {
            "time": max(crp.corpus_duration * rtf / crp.concurrent, 0.5),
            "cpu": 2,
            "gpu": 1 if self.use_gpu else 0,
            "mem": mem,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task(
            "run", resume="run", rqmt=self.rqmt, args=range(1, self.concurrent + 1)
        )

    def create_files(self):
        self.write_config(
            self.config, self.post_config, "create-raw-denominator.config"
        )
        self.feature_flow.write_to_file("feature.flow")
        util.write_paths_to_file(
            self.lattice_bundle, self.single_lattice_caches.values()
        )
        extra_code = (
            ":${{THEANO_FLAGS:="
            "}}\n"
            'export THEANO_FLAGS="$THEANO_FLAGS,device={0},force_device=True"\n'
            'export TF_DEVICE="{0}"'.format("gpu" if self.use_gpu else "cpu")
        )
        self.write_run_script(
            self.exe, "create-raw-denominator.config", extra_code=extra_code
        )

    def run(self, task_id):
        self.run_script(task_id, self.log_file[task_id])
        shutil.move(
            "raw-denominator.%d" % task_id,
            self.single_lattice_caches[task_id].get_path(),
        )

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("create-raw-denominator.log.%d" % task_id)
        util.delete_if_exists("raw-denominator.%d" % task_id)

    @classmethod
    def create_config(
        cls,
        crp,
        feature_flow,
        feature_scorer,
        search_parameters,
        lm_lookahead,
        lookahead_options,
        mem,
        model_combination_config,
        model_combination_post_config,
        extra_config,
        extra_post_config,
        **kwargs,
    ):

        lm_gc = recognition.AdvancedTreeSearchLmImageAndGlobalCacheJob(
            crp, feature_scorer, extra_config, extra_post_config
        )
        lm_gc.rqmt["mem"] = mem

        if search_parameters is None:
            search_parameters = {}

        default_search_parameters = {
            "beam-pruning": 15,
            "beam-pruning-limit": 100000,
            "word-end-pruning": 0.5,
            "word-end-pruning-limit": 10000,
        }
        default_search_parameters.update(search_parameters)
        search_parameters = default_search_parameters

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
                "corpus": "speech-recognizer.corpus",
                "lexicon": "speech-recognizer.model-combination.lexicon",
                "acoustic_model": "speech-recognizer.model-combination.acoustic-model",
                "language_model": "speech-recognizer.model-combination.lm",
            },
            parallelize=True,
        )

        # Parameters for Speech::Recognizer
        config.speech_recognizer.search_type = "advanced-tree-search"

        # Parameters for Speech::DataSource or Sparse::DataSource
        config.speech_recognizer.feature_extraction.file = "feature.flow"
        feature_flow.apply_config(
            "speech-recognizer.feature-extraction", config, post_config
        )

        # Parameters for Am::ClassicAcousticModel
        feature_scorer.apply_config(
            "speech-recognizer.model-combination.acoustic-model.mixture-set",
            config,
            post_config,
        )

        # Parameters for Speech::Model combination (besides AM and LM parameters)
        config.speech_recognizer.model_combination.pronunciation_scale = 3.0
        config.speech_recognizer.model_combination._update(model_combination_config)
        post_config.speech_recognizer.model_combination._update(
            model_combination_post_config
        )

        # Search parameters
        config.speech_recognizer.recognizer.create_lattice = True
        config.speech_recognizer.store_lattices = True

        config.speech_recognizer.recognizer.beam_pruning = search_parameters[
            "beam-pruning"
        ]
        config.speech_recognizer.recognizer.beam_pruning_limit = search_parameters[
            "beam-pruning-limit"
        ]
        config.speech_recognizer.recognizer.word_end_pruning = search_parameters[
            "word-end-pruning"
        ]
        config.speech_recognizer.recognizer.word_end_pruning_limit = search_parameters[
            "word-end-pruning-limit"
        ]

        config.speech_recognizer.recognizer.lm_lookahead = rasr.RasrConfig()
        config.speech_recognizer.recognizer.lm_lookahead._value = lm_lookahead
        config.speech_recognizer.recognizer.optimize_lattice = "simple"
        if lm_lookahead:
            config.speech_recognizer.recognizer.lm_lookahead_laziness = la_opts[
                "laziness"
            ]
            config.speech_recognizer.recognizer.lm_lookahead.history_limit = la_opts[
                "history_limit"
            ]
            config.speech_recognizer.recognizer.lm_lookahead.tree_cutoff = la_opts[
                "tree_cutoff"
            ]
            config.speech_recognizer.recognizer.lm_lookahead.minimum_representation = (
                la_opts["minimum_representation"]
            )
            post_config.speech_recognizer.recognizer.lm_lookahead.cache_size_low = (
                la_opts["cache_low"]
            )
            post_config.speech_recognizer.recognizer.lm_lookahead.cache_size_high = (
                la_opts["cache_high"]
            )

        post_config.speech_recognizer.global_cache.read_only = True
        post_config.speech_recognizer.global_cache.file = lm_gc.out_global_cache
        post_config.speech_recognizer.model_combination.lm.image = lm_gc.lm_image

        # Lattice writer options
        config.speech_recognizer.lattice_archive.path = "raw-denominator.$(TASK)"
        post_config.speech_recognizer.lattice_archive.info = True

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
                "exe": kwargs["crp"].speech_recognizer_exe,
            }
        )


class DenominatorLatticeJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        raw_denominator_path,
        numerator_path,
        use_gpu=False,
        rtf=1,
        mem=4,  # TODO check requirements
        search_options=None,
        extra_config=None,
        extra_post_config=None,
    ):

        self.set_vis_name("Denominator Lattice")

        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = self.create_config(**kwargs)
        self.exe = self.select_exe(crp.lattice_processor_exe, "lattice-processor")
        self.concurrent = crp.concurrent
        self.use_gpu = use_gpu

        self.log_file = self.log_file_output_path("create-denominator", crp, True)
        self.single_lattice_caches = {
            task_id: self.output_path("denominator.%d" % task_id, cached=True)
            for task_id in range(1, crp.concurrent + 1)
        }
        self.lattice_bundle = self.output_path("denominator.bundle", cached=True)
        self.lattice_path = util.MultiOutputPath(
            self, "denominator.$(TASK)", self.single_lattice_caches, cached=True
        )

        self.rqmt = {
            "time": max(crp.corpus_duration * rtf / crp.concurrent, 0.5),
            "cpu": 2,
            "gpu": 1 if self.use_gpu else 0,
            "mem": mem,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task(
            "run", resume="run", rqmt=self.rqmt, args=range(1, self.concurrent + 1)
        )

    def create_files(self):
        self.write_config(self.config, self.post_config, "create-denominator.config")
        util.write_paths_to_file(
            self.lattice_bundle, self.single_lattice_caches.values()
        )
        extra_code = (
            ":${{THEANO_FLAGS:="
            "}}\n"
            'export THEANO_FLAGS="$THEANO_FLAGS,device={0},force_device=True"\n'
            'export TF_DEVICE="{0}"'.format("gpu" if self.use_gpu else "cpu")
        )
        self.write_run_script(
            self.exe, "create-denominator.config", extra_code=extra_code
        )

    def run(self, task_id):
        self.run_script(task_id, self.log_file[task_id])
        shutil.move(
            "denominator.%d" % task_id, self.single_lattice_caches[task_id].get_path()
        )

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("create-denominator.log.%d" % task_id)
        util.delete_if_exists("denominator.%d" % task_id)

    @classmethod
    def create_config(
        cls,
        crp,
        raw_denominator_path,
        numerator_path,
        mem,
        search_options,
        extra_config,
        extra_post_config,
        **kwargs,
    ):

        search_opts = {"pruning-threshold": 15, "pruning-threshold-relative": True}

        if search_options:
            search_opts.update(search_options)

        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "corpus": "lattice-processor.corpus",
                "lexicon": [
                    "lattice-processor.lattice-reader.model-combination.lexicon",
                    "lattice-processor.merging.model-combination.lexicon",
                ],
                "acoustic_model": "lattice-processor.merging.acoustic-model",
                "language_model": [
                    "lattice-processor.lattice-reader.lm",
                    "lattice-processor.merging.lm",
                ],
            },
            parallelize=True,
        )

        # Define and name actions
        config.lattice_processor.actions = (
            "read,linear-combination,prune,graph-error-rate,merge,write"
        )
        config.lattice_processor.selections = (
            "lattice-reader,linear-combination,pruning,ger,merging,lattice-writer"
        )

        # Reader
        config.lattice_processor.lattice_reader.readers = "acoustic,lm"
        config.lattice_processor.lattice_reader.lattice_archive.path = (
            raw_denominator_path
        )

        # linear-combination
        config["*"].LM_SCALE = crp.language_model_config.scale
        config.lattice_processor.linear_combination.scales = ["$[1.0/$(LM-SCALE)]"] * 2

        # pruning
        config.lattice_processor.pruning.threshold = search_opts["pruning-threshold"]
        config.lattice_processor.pruning.threshold_is_relative = search_opts[
            "pruning-threshold-relative"
        ]

        # merging
        config.lattice_processor.merging.fsa_prefix = "acoustic"
        config.lattice_processor.merging.merge_only_if_spoken_not_in_lattice = True
        config.lattice_processor.merging.numerator_lattice_archive.path = numerator_path

        # graph error rate [needs no config]

        # writer
        config.lattice_processor.lattice_writer.lattice_archive.path = (
            "denominator.$(TASK)"
        )

        # additional config
        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        return super().hash(
            {"config": config, "exe": kwargs["crp"].lattice_processor_exe}
        )


class AccuracyJob(rasr.RasrCommand, Job):
    """This class is just a base class. Use StateAccuracyJob or PhoneAccuracyJob instead."""

    def __init__(
        self,
        crp,
        feature_flow,
        feature_scorer,
        denominator_path,
        alignment_options=None,
        short_pauses=None,
        use_gpu=False,
        rtf=40,
        mem=4,  # TODO check requirements
        extra_config=None,
        extra_post_config=None,
    ):

        self.set_vis_name("Accuracy Lattice")

        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = self.create_config(**kwargs)
        self.alignment_flow = self.create_flow(**kwargs)
        self.exe = self.select_exe(crp.lattice_processor_exe, "lattice-processor")
        self.concurrent = crp.concurrent
        self.use_gpu = use_gpu

        self.log_file = self.log_file_output_path("create-accuracy", crp, True)
        self.single_lattice_caches = {
            task_id: self.output_path("accuracy.%d" % task_id, cached=True)
            for task_id in range(1, crp.concurrent + 1)
        }
        self.lattice_bundle = self.output_path("accuracy.bundle", cached=True)
        self.lattice_path = util.MultiOutputPath(
            self, "accuracy.$(TASK)", self.single_lattice_caches, cached=True
        )
        self.single_segmentwise_alignment_caches = {
            task_id: self.output_path("segmentwise-alignment.%d" % task_id, cached=True)
            for task_id in range(1, crp.concurrent + 1)
        }
        self.segmentwise_alignment_bundle = self.output_path(
            "segmentwise-alignment.bundle", cached=True
        )
        self.segmentwise_alignment_path = util.MultiOutputPath(
            self,
            "segmentwise-alignment.$(TASK)",
            self.single_segmentwise_alignment_caches,
            cached=True,
        )

        self.rqmt = {
            "time": max(crp.corpus_duration * rtf / crp.concurrent, 0.5),
            "cpu": 2,
            "gpu": 1 if self.use_gpu else 0,
            "mem": mem,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task(
            "run", resume="run", rqmt=self.rqmt, args=range(1, self.concurrent + 1)
        )

    def create_files(self):
        self.write_config(self.config, self.post_config, "create-accuracy.config")
        self.alignment_flow.write_to_file("alignment.flow")
        util.write_paths_to_file(
            self.lattice_bundle, self.single_lattice_caches.values()
        )
        util.write_paths_to_file(
            self.segmentwise_alignment_bundle,
            self.single_segmentwise_alignment_caches.values(),
        )
        extra_code = (
            ":${{THEANO_FLAGS:="
            "}}\n"
            'export THEANO_FLAGS="$THEANO_FLAGS,device={0},force_device=True"\n'
            'export TF_DEVICE="{0}"'.format("gpu" if self.use_gpu else "cpu")
        )
        self.write_run_script(self.exe, "create-accuracy.config", extra_code=extra_code)

    def run(self, task_id):
        self.run_script(task_id, self.log_file[task_id])
        shutil.move(
            "accuracy.%d" % task_id, self.single_lattice_caches[task_id].get_path()
        )
        shutil.move(
            "segmentwise-alignment.%d" % task_id,
            self.single_segmentwise_alignment_caches[task_id].get_path(),
        )

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("create-accuracy.log.%d" % task_id)
        util.delete_if_exists("accuracy.%d" % task_id)
        util.delete_if_exists("segmentwise-alignment.%d" % task_id)

    @classmethod
    def create_config(
        cls,
        crp,
        feature_flow,
        feature_scorer,
        denominator_path,
        mem,
        alignment_options,
        short_pauses,
        extra_config,
        extra_post_config,
        **kwargs,
    ):

        alignment_flow = cls.create_flow(feature_flow)

        alignopt = {
            "increase-pruning-until-no-score-difference": True,
            "min-acoustic-pruning": 1000,
            "max-acoustic-pruning": 512000,
            "acoustic-pruning-increment-factor": 2,
            "min-average-number-of-states": 12,
        }

        if alignment_options is not None:
            alignopt.update(alignment_options)

        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "corpus": "lattice-processor.corpus",
                "lexicon": [
                    "lattice-processor.topology-reader.model-combination.lexicon",
                    "lattice-processor.rescoring.tdps.model-combination.lexicon",
                    "lattice-processor.rescoring.segmentwise-alignment.model-combination.lexicon",
                ],
                "acoustic_model": [
                    "lattice-processor.rescoring.tdps.model-combination.acoustic-model",
                    "lattice-processor.rescoring.segmentwise-alignment.model-combination.acoustic-model",
                ],
            },
            parallelize=True,
        )
        # Define and name actions
        config.lattice_processor.actions = "read,rescore,write"
        config.lattice_processor.selections = (
            "topology-reader,rescoring,accuracy-writer"
        )

        # Reader
        config.lattice_processor.topology_reader.readers = "total"
        config.lattice_processor.topology_reader.lattice_archive.path = denominator_path

        # rescoring
        config.lattice_processor.rescoring.share_acoustic_model = True
        config.lattice_processor.rescoring.tdp_rescorers = "tdps"
        config.lattice_processor.rescoring.distance_rescorers = "accuracy"
        config.lattice_processor.rescoring.accuracy.spoken_source = "orthography"

        # Parameters for Am::ClassicAcousticModel
        for node in ["tdps", "segmentwise-alignment"]:
            feature_scorer.apply_config(
                "lattice-processor.rescoring.{}.model-combination.acoustic-model.mixture-set".format(
                    node
                ),
                config,
                post_config,
            )

        # rescoring aligner
        config.lattice_processor.rescoring.segmentwise_alignment.port_name = "features"
        config.lattice_processor.rescoring.segmentwise_alignment.alignment_cache.alignment_label_type = (
            "emission-ids"
        )
        config.lattice_processor.rescoring.segmentwise_alignment.alignment_cache.path = (
            "segmentwise-alignment.$(TASK)"
        )
        post_config.lattice_processor.rescoring.segmentwise_alignment.model_acceptor_cache.log.channel = (
            "nil"
        )
        post_config.lattice_processor.rescoring.segmentwise_alignment.aligner.statistics.channel = (
            "nil"
        )
        node = config.lattice_processor.rescoring.segmentwise_alignment.aligner
        for k, v in alignopt.items():
            node[k] = v

        alignment_flow.apply_config(
            "lattice-processor.rescoring.segmentwise-feature-extraction.feature-extraction",
            config,
            post_config,
        )
        config.lattice_processor.rescoring.segmentwise_feature_extraction.feature_extraction.file = (
            "alignment.flow"
        )

        # writer
        config.lattice_processor.accuracy_writer.lattice_archive.path = (
            "accuracy.$(TASK)"
        )

        return config, post_config

    @classmethod
    def create_flow(cls, feature_flow, **kwargs):
        flow = alignment_flow(feature_flow, None)
        return flow

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        return super().hash(
            {
                "config": config,
                "feature_flow": kwargs["feature_flow"],
                "exe": kwargs["crp"].lattice_processor_exe,
            }
        )


class PhoneAccuracyJob(AccuracyJob, Job):
    def __init__(
        self,
        crp,
        feature_flow,
        feature_scorer,
        denominator_path,
        alignment_options=None,
        short_pauses=None,
        use_gpu=False,
        rtf=40,
        mem=8,  # TODO check requirements
        extra_config=None,
        extra_post_config=None,
    ):
        """see AccuracyJob for list of kwargs"""
        super().__init__(
            crp,
            feature_flow,
            feature_scorer,
            denominator_path,
            alignment_options,
            short_pauses,
            use_gpu,
            rtf,
            mem,
            extra_config,
            extra_post_config,
        )
        self.set_vis_name("Phone Accuracy Lattice")

    @classmethod
    def create_config(
        cls,
        crp,
        feature_flow,
        feature_scorer,
        denominator_path,
        mem,
        alignment_options,
        short_pauses,
        extra_config,
        extra_post_config,
        **kwargs,
    ):

        config, post_config = super().create_config(
            crp,
            feature_flow,
            feature_scorer,
            denominator_path,
            mem,
            alignment_options,
            short_pauses,
            extra_config,
            extra_post_config,
            **kwargs,
        )

        config.lattice_processor.rescoring.accuracy.distance_type = (
            "approximate-phone-accuracy"
        )
        config.lattice_processor.rescoring.accuracy.approximate_phone_accuracy_lattice_builder.token_type = (
            "phone"
        )

        if short_pauses is not None:
            config.lattice_processor.rescoring.accuracy.approximate_phone_accuracy_lattice_builder.short_pauses_lemmata = " ".join(
                short_pauses
            )

        # additional config
        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config


class StateAccuracyJob(AccuracyJob, Job):
    def __init__(
        self,
        crp,
        feature_flow,
        feature_scorer,
        denominator_path,
        alignment_options=None,
        short_pauses=None,
        use_gpu=False,
        rtf=40,
        mem=8,  # TODO check requirements
        extra_config=None,
        extra_post_config=None,
    ):
        """see AccuracyJob for list of kwargs"""
        super().__init__(
            crp,
            feature_flow,
            feature_scorer,
            denominator_path,
            alignment_options,
            short_pauses,
            use_gpu,
            rtf,
            mem,
            extra_config,
            extra_post_config,
        )
        self.set_vis_name("State Accuracy Lattice")

    @classmethod
    def create_config(
        cls,
        crp,
        feature_flow,
        feature_scorer,
        denominator_path,
        mem,
        alignment_options,
        short_pauses,
        extra_config,
        extra_post_config,
        **kwargs,
    ):

        config, post_config = super().create_config(
            crp,
            feature_flow,
            feature_scorer,
            denominator_path,
            mem,
            alignment_options,
            short_pauses,
            extra_config,
            extra_post_config,
            **kwargs,
        )

        config.lattice_processor.rescoring.accuracy.distance_type = (
            "smoothed-frame-state-accuracy"
        )
        config.lattice_processor.rescoring.accuracy.approximate_phone_accuracy_lattice_builder.token_type = (
            "state"
        )

        # additional config
        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config
