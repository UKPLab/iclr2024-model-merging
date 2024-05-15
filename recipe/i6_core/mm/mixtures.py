__all__ = [
    "MergeMixturesJob",
    "LinearAlignmentJob",
    "EstimateMixturesJob",
    "CreateDummyMixturesJob",
]

import logging
import os
import shutil
import stat
import struct
import tempfile

from sisyphus import *

Path = setup_path(__package__)

from .flow import linear_segmentation_flow, cached_alignment_flow
import i6_core.rasr as rasr
import i6_core.util as util


class MergeMixturesJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        mixtures_to_combine,
        combine_per_step=20,
        estimator="maximum-likelihood",
        extra_config=None,
        extra_post_config=None,
    ):
        (
            self.config_merge,
            self.post_config_merge,
        ) = MergeMixturesJob.create_merge_config(
            crp, estimator, extra_config, extra_post_config
        )
        self.merge_exe = (
            crp.acoustic_model_trainer_exe
            if crp.acoustic_model_trainer_exe is not None
            else self.default_exe("acoustic-model-trainer")
        )
        if type(mixtures_to_combine) == dict:
            mixtures_to_combine = [
                mixtures_to_combine[k] for k in sorted(mixtures_to_combine)
            ]
        self.mixtures_to_combine = mixtures_to_combine
        self.combine_per_step = combine_per_step

        # determine how many merges have to be done
        merge_count = 0

        def inc_merge_count(e):
            nonlocal merge_count
            merge_count += 1

        util.reduce_tree(
            inc_merge_count,
            util.partition_into_tree(mixtures_to_combine, combine_per_step),
        )

        self.out_merge_log_file = self.log_file_output_path("merge", crp, merge_count)
        self.out_mixtures = self.output_path("am.mix", cached=True)

        self.merge_rqmt = {"time": max(merge_count / 25, 0.5), "cpu": 1, "mem": 1}

    def tasks(self):
        yield Task("create_merge_mixtures_config", mini_task=True)
        yield Task("merge_mixtures", rqmt=self.merge_rqmt)

    def create_merge_mixtures_config(self):
        with open("merge-mixtures.config", "wt") as f:
            self.config_merge._update(self.post_config_merge)
            f.write(repr(self.config_merge))

    def merge_mixtures(self):
        merge_num = 0
        tmp_files = set()

        def merge_helper(elements):
            nonlocal merge_num
            merge_num += 1

            (fd, tmp_merge_file) = tempfile.mkstemp(suffix=".mix")
            os.close(fd)
            logging.info("merge %d, %r -> %s", merge_num, elements, tmp_merge_file)

            tmp_files.add(tmp_merge_file)

            self.run_cmd(
                self.merge_exe,
                [
                    "--config=merge-mixtures.config",
                    "--*.TASK=1",
                    "--*.LOGFILE=merge.log.%d" % merge_num,
                    "--mixture-set-trainer.mixture-set-files-to-combine=%s"
                    % " ".join(elements),
                    "--mixture-set-trainer.new-mixture-set-file=%s" % tmp_merge_file,
                ],
            )
            util.zmove(
                "merge.log.%d" % merge_num,
                self.out_merge_log_file[merge_num].get_path(),
            )

            for e in elements:
                if e in tmp_files:
                    logging.info("unlink %s" % e)
                    os.unlink(e)
                    tmp_files.remove(e)

            return tmp_merge_file

        mixtures = util.reduce_tree(
            merge_helper,
            util.partition_into_tree(self.mixtures_to_combine, self.combine_per_step),
        )
        shutil.move(mixtures, self.out_mixtures.get_path())

        update_mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IROTH | stat.S_IRGRP
        os.chmod(self.out_mixtures.get_path(), update_mode)

    def cleanup_before_run(self, cmd, retry, *args):
        log = args[2][12:]
        util.backup_if_exists(log)

    @classmethod
    def create_merge_config(
        cls, crp, estimator, extra_config, extra_post_config, **kwargs
    ):
        config, post_config = rasr.build_config_from_mapping(crp, {})
        config.acoustic_model_trainer.action = "combine-mixture-sets"
        config.acoustic_model_trainer.mixture_set_trainer.estimator_type = estimator

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_merge_config(**kwargs)
        return Job.hash(
            {
                "config": config,
                "mixtures": kwargs["mixtures_to_combine"],
                "combine_per_step": kwargs["combine_per_step"],
                "exe": kwargs["crp"].acoustic_model_trainer_exe,
            }
        )


class LinearAlignmentJob(MergeMixturesJob):
    def __init__(
        self,
        crp,
        feature_energy_flow,
        minimum_segment_length=0,
        maximum_segment_length=6000,
        iterations=1,
        penalty=0,
        minimum_speech_proportion=0.7,
        save_alignment=False,
        keep_accumulators=False,
        extra_merge_args=None,
        extra_config=None,
        extra_post_config=None,
    ):
        self.set_vis_name("Linear Alignment")

        kwargs = locals()
        del kwargs["self"]

        super().__init__(**LinearAlignmentJob.merge_args(**kwargs))

        self.config, self.post_config = LinearAlignmentJob.create_config(**kwargs)
        self.linear_alignment_flow = LinearAlignmentJob.create_flow(**kwargs)
        self.exe = self.select_exe(
            crp.acoustic_model_trainer_exe, "acoustic-model-trainer"
        )
        self.concurrent = crp.concurrent
        self.save_alignment = save_alignment
        self.keep_accumulators = keep_accumulators

        self.out_log_file = self.log_file_output_path("accumulate", crp, True)
        if save_alignment:
            self.single_alignment_caches = dict(
                (i, self.output_path("alignment.cache.%d" % i, cached=True))
                for i in range(1, self.concurrent + 1)
            )
            self.out_alignment_path = util.MultiOutputPath(
                self,
                "alignment.cache.$(TASK)",
                self.single_alignment_caches,
                cached=True,
            )
            self.out_alignment_bundle = self.output_path("alignment.cache.bundle")

        self.accumulate_rqmt = {
            "time": max(crp.corpus_duration / (20.0 * self.concurrent), 0.5),
            "cpu": 1,
            "mem": 1,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task(
            "accumulate",
            resume="accumulate",
            rqmt=self.accumulate_rqmt,
            args=range(1, self.concurrent + 1),
        )
        yield from super().tasks()
        if not self.keep_accumulators:
            yield Task("delete_accumulators", mini_task=True)

    def create_files(self):
        self.write_config(self.config, self.post_config, "linear-segmentation.config")
        self.linear_alignment_flow.write_to_file("linear-alignment.flow")
        self.write_run_script(self.exe, "linear-segmentation.config", "accumulate.sh")
        if self.save_alignment:
            util.write_paths_to_file(
                self.out_alignment_bundle, self.single_alignment_caches.values()
            )

    def accumulate(self, task_id):
        self.run_script(task_id, self.out_log_file[task_id], "./accumulate.sh")
        if self.save_alignment:
            shutil.move(
                "alignment.cache.%d" % task_id,
                self.single_alignment_caches[task_id].get_path(),
            )

    def delete_accumulators(self):
        for i in range(1, self.concurrent + 1):
            if os.path.exists("linear.acc.%d" % i):
                os.remove("linear.acc.%d" % i)

    def cleanup_before_run(self, cmd, retry, *args):
        if cmd == self.merge_exe:
            super().cleanup_before_run(cmd, retry, *args)
        else:
            task_id = args[0]
            util.backup_if_exists("accumulate.log.%d" % task_id)
            if self.save_alignment:
                util.delete_if_zero("alignment.cache.%d" % task_id)

    @classmethod
    def create_config(
        cls,
        crp,
        feature_energy_flow,
        minimum_segment_length,
        maximum_segment_length,
        iterations,
        penalty,
        minimum_speech_proportion,
        save_alignment,
        extra_config,
        extra_post_config,
        **kwargs,
    ):
        segmentation_flow = cls.create_flow(feature_energy_flow, save_alignment)
        mapping = {
            "corpus": "acoustic-model-trainer.corpus",
            "lexicon": ["acoustic-model-trainer.mixture-set-trainer.lexicon"],
            "acoustic_model": [
                "acoustic-model-trainer.mixture-set-trainer.acoustic-model"
            ],
        }

        # acoustic model + lexicon for the flow nodes
        for node in segmentation_flow.get_node_names_by_filter(
            "speech-linear-segmentation"
        ):
            node_path = (
                "acoustic-model-trainer.aligning-feature-extractor.feature-extraction."
                + node
            )
            mapping["lexicon"].append("%s.model-combination.lexicon" % node_path)
            mapping["acoustic_model"].append(
                "%s.model-combination.acoustic-model" % node_path
            )

        config, post_config = rasr.build_config_from_mapping(
            crp, mapping, parallelize=True
        )

        # shortcuts
        fe = config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction
        mst = config.acoustic_model_trainer.mixture_set_trainer

        # segmentation options for the flow nodes
        for node in segmentation_flow.get_node_names_by_filter(
            "speech-linear-segmentation"
        ):
            fe[node].linear_segmenter.minimum_segment_length = minimum_segment_length
            fe[node].linear_segmenter.maximum_segment_length = maximum_segment_length
            fe[node].linear_segmenter.delimiter.number_of_iterations = iterations
            fe[node].linear_segmenter.delimiter.penalty = penalty
            fe[
                node
            ].linear_segmenter.delimiter.minimum_speech_proportion = (
                minimum_speech_proportion
            )

        config.action = "accumulate-mixture-set-text-dependent"
        mst.new_mixture_set_file = "linear.acc.$(TASK)"
        fe.file = "linear-alignment.flow"

        segmentation_flow.apply_config(
            "acoustic-model-trainer.aligning-feature-extractor.feature-extraction",
            config,
            post_config,
        )

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def create_flow(cls, feature_energy_flow, save_alignment, **kwargs):
        return linear_segmentation_flow(
            feature_energy_flow, "alignment.cache.$(TASK)" if save_alignment else None
        )

    @classmethod
    def merge_args(cls, crp, extra_merge_args, **kwargs):
        merge_args = {
            "crp": crp,
            "mixtures_to_combine": [
                "linear.acc.%d" % i for i in range(1, crp.concurrent + 1)
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
        linear_alignment_flow = cls.create_flow(**kwargs)
        return Job.hash(
            {
                "config": config,
                "linear_alignment_flow": linear_alignment_flow,
                "exe": kwargs["crp"].acoustic_model_trainer_exe,
                "merge_hash": MergeMixturesJob.hash(cls.merge_args(**kwargs)),
            }
        )


class EstimateMixturesJob(MergeMixturesJob):
    def __init__(
        self,
        crp,
        old_mixtures,
        feature_flow,
        alignment,
        split_first=True,
        keep_accumulators=False,
        extra_merge_args=None,
        extra_config=None,
        extra_post_config=None,
    ):
        self.set_vis_name("Split Mixtures" if split_first else "Accumulate Mixtures")

        kwargs = locals()
        del kwargs["self"]
        super().__init__(**EstimateMixturesJob.merge_args(**kwargs))

        self.config, self.post_config = EstimateMixturesJob.create_config(**kwargs)
        self.alignment_flow = EstimateMixturesJob.create_flow(**kwargs)
        self.exe = self.select_exe(
            crp.acoustic_model_trainer_exe, "acoustic-model-trainer"
        )
        self.split_first = split_first
        self.keep_accumulators = keep_accumulators
        self.concurrent = crp.concurrent

        self._old_mixtures = old_mixtures

        self.out_log_file = self.log_file_output_path("accumulate", crp, True)

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
        self.alignment_flow.write_to_file("alignment.flow")
        self.write_run_script(self.exe, "accumulate-mixtures.config", "accumulate.sh")

    def accumulate(self, task_id):
        self.run_script(task_id, self.out_log_file[task_id], "./accumulate.sh")

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
        alignment,
        split_first,
        extra_config,
        extra_post_config,
        **kwargs,
    ):
        alignment_flow = cls.create_flow(feature_flow, alignment)
        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "corpus": "acoustic-model-trainer.corpus",
                "lexicon": "acoustic-model-trainer.mixture-set-trainer.lexicon",
                "acoustic_model": "acoustic-model-trainer.mixture-set-trainer.acoustic-model",
            },
            parallelize=True,
        )

        config.acoustic_model_trainer.action = "accumulate-mixture-set-text-dependent"
        config.acoustic_model_trainer.mixture_set_trainer.split_first = split_first
        config.acoustic_model_trainer.mixture_set_trainer.old_mixture_set_file = (
            old_mixtures
        )
        config.acoustic_model_trainer.mixture_set_trainer.new_mixture_set_file = (
            "am.acc.$(TASK)"
        )
        config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction.file = (
            "alignment.flow"
        )

        alignment_flow.apply_config(
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
                "merge_hash": MergeMixturesJob.hash(cls.merge_args(**kwargs)),
            }
        )


class CreateDummyMixturesJob(Job):
    def __init__(self, num_mixtures, num_features):
        self.num_mixtures = num_mixtures
        self.num_features = num_features

        self.out_mixtures = self.output_path("dummy.mix")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        num_mixtures = int(
            self.num_mixtures.get()
            if isinstance(self.num_mixtures, tk.Variable)
            else self.num_mixtures
        )
        num_features = int(
            self.num_features.get()
            if isinstance(self.num_features, tk.Variable)
            else self.num_features
        )

        with open(tk.uncached_path(self.out_mixtures), "wb") as f:
            f.write(b"MIXSET\0\0")
            f.write(struct.pack("II", 2, num_features))
            args = [1, num_features] + [0.0] * num_features + [1.0]
            f.write(struct.pack("II%ddd" % num_features, *args))  # mean accumulator
            f.write(struct.pack("II%ddd" % num_features, *args))  # var  accumulator
            f.write(
                struct.pack("IIII", 1, 0, 0, num_mixtures)
            )  # num density + density mean/var idx + num of mixtures
            single_mixture = struct.pack("IId", 1, 0, 1.0)
            for i in range(num_mixtures):
                f.write(single_mixture)
