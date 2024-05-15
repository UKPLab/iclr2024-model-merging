__all__ = ["EstimateScatterMatricesJob", "EstimateLDAMatrixJob"]

import os
import shutil
import tempfile

from sisyphus import *
from sisyphus.delayed_ops import DelayedAdd

Path = setup_path(__package__)

import i6_core.rasr as rasr
import i6_core.util as util


class EstimateScatterMatricesJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        alignment_flow,
        combine_per_step=20,
        keep_accumulators=False,
        extra_config_accumulate=None,
        extra_post_config_accumulate=None,
        extra_config_merge=None,
        extra_post_config_merge=None,
        extra_config_estimate=None,
        extra_post_config_estimate=None,
    ):
        self.set_vis_name("Estimate Scatter Matrices")

        kwargs = locals()
        del kwargs["self"]

        self.alignment_flow = alignment_flow
        self.combine_per_step = combine_per_step
        self.concurrent = crp.concurrent
        (
            self.config_accumulate,
            self.post_config_accumulate,
        ) = EstimateScatterMatricesJob.create_accumulate_config(**kwargs)
        (
            self.config_merge,
            self.post_config_merge,
        ) = EstimateScatterMatricesJob.create_merge_config(**kwargs)
        (
            self.config_estimate,
            self.post_config_estimate,
        ) = EstimateScatterMatricesJob.create_estimate_config(**kwargs)
        self.exe = self.select_exe(
            crp.acoustic_model_trainer_exe, "acoustic-model-trainer"
        )
        self.keep_accumulators = keep_accumulators

        # determine how many merges have to be done
        merge_count = 0

        def inc_merge_count(e):
            nonlocal merge_count
            merge_count += 1

        util.reduce_tree(
            inc_merge_count,
            util.partition_into_tree(list(range(self.concurrent)), combine_per_step),
        )

        self.accumulate_log_file = self.log_file_output_path("accumulate", crp, True)
        self.merge_log_file = self.log_file_output_path("merge", crp, merge_count)
        self.estimate_log_file = self.log_file_output_path("estimate", crp, False)
        self.between_class_scatter_matrix = self.output_path(
            "between_class_scatter.matrix"
        )
        self.within_class_scatter_matrix = self.output_path(
            "within_class_scatter.matrix"
        )
        self.total_scatter_matrix = self.output_path("total_scatter.matrix")

        self.accumulate_rqmt = {
            "time": max(crp.corpus_duration / (20 * crp.concurrent), 0.5),
            "cpu": 1,
            "mem": 4,
        }
        self.merge_rqmt = {"time": 0.5, "cpu": 1, "mem": 1}

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task(
            "accumulate",
            resume="accumulate",
            rqmt=self.accumulate_rqmt,
            args=range(1, self.concurrent + 1),
        )
        yield Task("merge", resume="merge", rqmt=self.merge_rqmt)

    def create_files(self):
        self.alignment_flow.write_to_file("alignment.flow")
        self.write_config(
            self.config_accumulate, self.post_config_accumulate, "accumulate.config"
        )
        self.write_config(self.config_merge, self.post_config_merge, "merge.config")
        self.write_config(
            self.config_estimate, self.post_config_estimate, "estimate.config"
        )
        self.write_run_script(self.exe, "accumulate.config", "accumulate.sh")
        self.write_run_script(self.exe, "estimate.config", "estimate.sh")

    def accumulate(self, task_id):
        self.run_script(task_id, self.accumulate_log_file[task_id], "./accumulate.sh")

    def merge(self):
        merge_num = 0
        tmp_files_to_delete = []

        def merge_helper(elements):
            nonlocal merge_num
            merge_num += 1

            (fd, tmp_merge_file) = tempfile.mkstemp(suffix=".acc")
            os.close(fd)

            self.run_cmd(
                self.exe,
                [
                    "--config=merge.config",
                    "--*.TASK=1",
                    "--*.LOGFILE=merge.log.%d" % merge_num,
                    "--*.scatter-matrix-estimator.accumulator-files-to-combine=%s"
                    % " ".join(elements),
                    "--*.scatter-matrix-estimator.new-accumulator-file=%s"
                    % tmp_merge_file,
                ],
            )
            util.zmove(
                "merge.log.%d" % merge_num, self.merge_log_file[merge_num].get_path()
            )

            return tmp_merge_file

        final_accumulator = util.reduce_tree(
            merge_helper,
            util.partition_into_tree(
                ["scatter.acc.%d" % i for i in range(1, self.concurrent + 1)],
                self.combine_per_step,
            ),
        )

        self.run_script(
            1,
            self.estimate_log_file,
            "./estimate.sh",
            [
                "--*.scatter-matrix-estimator.old-accumulator-file=%s"
                % final_accumulator
            ],
        )
        shutil.move(
            "between_class_scatter.matrix", self.between_class_scatter_matrix.get_path()
        )
        shutil.move(
            "within_class_scatter.matrix", self.within_class_scatter_matrix.get_path()
        )
        shutil.move("total_scatter.matrix", self.total_scatter_matrix.get_path())

        for tmp_file in tmp_files_to_delete:
            os.remove(tmp_file)
        if not self.keep_accumulators:
            for i in range(1, self.concurrent + 1):
                os.remove("scatter.acc.%d" % i)

    def cleanup_before_run(self, cmd, retry, *args):
        if cmd == "./accumulate.sh":
            task_id = args[0]
            util.backup_if_exists("accumulate.log.%d" % task_id)
        elif cmd == self.exe:
            log_file = args[2][12:]
            util.backup_if_exists(log_file)
        elif cmd == "./estimate.sh":
            util.backup_if_exists("estimate.log")

    @classmethod
    def create_accumulate_config(
        cls,
        crp,
        alignment_flow,
        extra_config_accumulate,
        extra_post_config_accumulate,
        **kwargs,
    ):
        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "corpus": "acoustic-model-trainer.corpus",
                "lexicon": "acoustic-model-trainer.scatter-matrices-estimator.lexicon",
                "acoustic_model": "acoustic-model-trainer.scatter-matrices-estimator.acoustic-model",
            },
            parallelize=True,
        )

        config.acoustic_model_trainer.action = (
            "estimate-scatter-matrices-text-dependent"
        )
        config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction.file = (
            "alignment.flow"
        )
        config.acoustic_model_trainer.scatter_matrices_estimator.new_accumulator_file = (
            "scatter.acc.$(TASK)"
        )

        alignment_flow.apply_config(
            "acoustic-model-trainer.aligning-feature-extractor.feature-extraction",
            config,
            post_config,
        )

        config._update(extra_config_accumulate)
        post_config._update(extra_post_config_accumulate)

        return config, post_config

    @classmethod
    def create_merge_config(
        cls, crp, extra_config_merge, extra_post_config_merge, **kwargs
    ):
        config, post_config = rasr.build_config_from_mapping(crp, {})

        config.acoustic_model_trainer.action = "combine-scatter-matrix-accumulators"

        config._update(extra_config_merge)
        post_config._update(extra_post_config_merge)

        return config, post_config

    @classmethod
    def create_estimate_config(
        cls, crp, extra_config_estimate, extra_post_config_estimate, **kwargs
    ):
        config, post_config = rasr.build_config_from_mapping(crp, {})

        config.acoustic_model_trainer.action = (
            "estimate-scatter-matrices-from-accumulator"
        )
        config.acoustic_model_trainer.scatter_matrix_estimator.between_class_scatter_matrix_file = (
            "between_class_scatter.matrix"
        )
        config.acoustic_model_trainer.scatter_matrix_estimator.within_class_scatter_matrix_file = (
            "within_class_scatter.matrix"
        )
        config.acoustic_model_trainer.scatter_matrix_estimator.total_scatter_matrix_file = (
            "total_scatter.matrix"
        )
        config.acoustic_model_trainer.scatter_matrix_estimator.shall_normalize = True
        config.acoustic_model_trainer.scatter_matrix_estimator.output_precision = 20

        config._update(extra_config_estimate)
        post_config._update(extra_post_config_estimate)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config_accumulate, ignore = cls.create_accumulate_config(**kwargs)
        config_merge, ignore = cls.create_merge_config(**kwargs)
        return super().hash(
            {
                "config_accumulate": config_accumulate,
                "config_merge": config_merge,
                "alignment_flow": kwargs["alignment_flow"],
                "exe": kwargs["crp"].acoustic_model_trainer_exe,
            }
        )


class EstimateLDAMatrixJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        between_class_scatter_matrix,
        within_class_scatter_matrix,
        reduced_dimension,
        eigenvalue_problem_config,
        generalized_eigenvalue_problem_config,
        extra_config=None,
        extra_post_config=None,
    ):
        assert isinstance(eigenvalue_problem_config, rasr.RasrConfig)
        assert isinstance(generalized_eigenvalue_problem_config, rasr.RasrConfig)

        self.set_vis_name("Estimate LDA Matrix")

        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = EstimateLDAMatrixJob.create_config(**kwargs)
        self.exe = self.select_exe(
            crp.acoustic_model_trainer_exe, "acoustic-model-trainer"
        )

        self.log_file = self.log_file_output_path("lda", crp, False)
        self.lda_matrix = self.output_path("lda.matrix", cached=True)

        self.rqmt = {"time": 0.5, "cpu": 1, "mem": 1}
        self._additional_inputs = [
            between_class_scatter_matrix,
            within_class_scatter_matrix,
        ]

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def create_files(self):
        self.write_config(self.config, self.post_config, "lda.config")
        self.write_run_script(self.exe, "lda.config")

    def run(self):
        self.run_script(1, self.log_file)
        shutil.move("lda.matrix", self.lda_matrix.get_path())

    def cleanup_before_run(self, *args):
        util.backup_if_exists("lda.log")

    @classmethod
    def create_config(
        cls,
        crp,
        between_class_scatter_matrix,
        within_class_scatter_matrix,
        reduced_dimension,
        eigenvalue_problem_config,
        generalized_eigenvalue_problem_config,
        extra_config,
        extra_post_config,
    ):
        config, post_config = rasr.build_config_from_mapping(crp, {})

        config.acoustic_model_trainer.action = "estimate-lda"
        config.acoustic_model_trainer.lda_estimator.reduced_dimesion = (
            reduced_dimension  # yes, reduced-dimesion is the correct config parameter
        )
        config.acoustic_model_trainer.lda_estimator.output_precision = 20

        config.acoustic_model_trainer.lda_estimator.between_class_scatter_matrix_file = DelayedAdd(
            "xml:", between_class_scatter_matrix
        )
        config.acoustic_model_trainer.lda_estimator.within_class_scatter_matrix_file = (
            DelayedAdd("xml:", within_class_scatter_matrix)
        )
        config.acoustic_model_trainer.lda_estimator.projector_matrix_file = (
            "xml:lda.matrix"
        )

        config.acoustic_model_trainer.lda_estimator.results.channel = (
            crp.default_log_channel
        )
        config.acoustic_model_trainer.lda_estimator.generalized_eigenvalue_problem.condition_numbers.channel = (
            crp.default_log_channel
        )

        config.acoustic_model_trainer.lda_estimator.eigenvalue_problem = (
            eigenvalue_problem_config
        )
        config.acoustic_model_trainer.lda_estimator.generalized_eigenvalue_problem = (
            generalized_eigenvalue_problem_config
        )

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        return super().hash(
            {"config": config, "exe": kwargs["crp"].acoustic_model_trainer_exe}
        )
