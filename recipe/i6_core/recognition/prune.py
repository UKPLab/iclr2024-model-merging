__all__ = ["LatticePruningJob"]

from sisyphus import *

Path = setup_path(__package__)

import shutil

import i6_core.rasr as rasr
import i6_core.util as util


class LatticePruningJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        lattice_path,
        pruning_threshold=100,
        phone_coverage=0,
        nonword_phones="[*",
        max_arcs_per_second=50000,
        max_arcs_per_segment=1000000,
        output_format="flf",
        pronunciation_scale=None,
        extra_config=None,
        extra_post_config=None,
    ):
        self.set_vis_name("Lattice Pruning")

        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = self.create_config(**kwargs)
        self.exe = self.select_exe(crp.flf_tool_exe, "flf-tool")
        self.concurrent = crp.concurrent

        self.out_log_file = self.log_file_output_path("pruning", crp, True)
        self.out_single_lattice_caches = dict(
            (
                task_id,
                self.output_path("pruned_lattice.cache.%d" % task_id, cached=True),
            )
            for task_id in range(1, crp.concurrent + 1)
        )
        self.out_lattice_bundle = self.output_path("pruned_lattice.bundle", cached=True)
        self.out_lattice_path = util.MultiOutputPath(
            self,
            "pruned_lattice.cache.$(TASK)",
            self.out_single_lattice_caches,
            cached=True,
        )

        self.rqmt = {
            "time": max(crp.corpus_duration * 0.2 / crp.concurrent, 0.5),
            "cpu": 1,
            "gpu": 0,
            "mem": 2.0,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task(
            "run", resume="run", rqmt=self.rqmt, args=range(1, self.concurrent + 1)
        )

    def create_files(self):
        self.write_config(self.config, self.post_config, "pruning.config")
        util.write_paths_to_file(
            self.out_lattice_bundle, self.out_single_lattice_caches.values()
        )
        self.write_run_script(self.exe, "pruning.config")

    def run(self, task_id):
        self.run_script(task_id, self.out_log_file[task_id])
        shutil.move(
            "pruned_lattice.cache.%d" % task_id,
            self.out_single_lattice_caches[task_id].get_path(),
        )

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("pruning.log.%d" % task_id)
        util.delete_if_exists("pruned_lattice.cache.%d" % task_id)

    @classmethod
    def create_config(
        cls,
        crp,
        lattice_path,
        pruning_threshold,
        phone_coverage,
        nonword_phones,
        max_arcs_per_second,
        max_arcs_per_segment,
        output_format,
        pronunciation_scale,
        extra_config,
        extra_post_config,
    ):
        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "corpus": "flf-lattice-tool.corpus",
                "lexicon": "flf-lattice-tool.lexicon",
            },
            parallelize=True,
        )

        config.flf_lattice_tool.network.initial_nodes = "segment"

        config.flf_lattice_tool.network.segment.type = "speech-segment"
        config.flf_lattice_tool.network.segment.links = (
            "0->archive-reader:1 0->archive-writer:1"
        )

        config.flf_lattice_tool.network.archive_reader.type = "archive-reader"
        config.flf_lattice_tool.network.archive_reader.format = "flf"
        config.flf_lattice_tool.network.archive_reader.path = lattice_path
        config.flf_lattice_tool.network.archive_reader.links = "0->prune:0"

        config.flf_lattice_tool.network.prune.type = "prune-posterior"
        config.flf_lattice_tool.network.prune.threshold = pruning_threshold
        config.flf_lattice_tool.network.prune.nonword_phones = nonword_phones
        config.flf_lattice_tool.network.prune.min_phone_coverage = phone_coverage
        if max_arcs_per_second is not None:
            config.flf_lattice_tool.network.prune.max_arcs_per_second = (
                max_arcs_per_second
            )
        if max_arcs_per_segment is not None:
            config.flf_lattice_tool.network.prune.max_arcs_per_segment = (
                max_arcs_per_segment
            )
        config.flf_lattice_tool.network.prune.links = "apply-pruning"

        config.flf_lattice_tool.network.apply_pruning.type = "copy"
        config.flf_lattice_tool.network.apply_pruning.trim = True
        config.flf_lattice_tool.network.apply_pruning.normalize = True
        config.flf_lattice_tool.network.apply_pruning.info = True
        config.flf_lattice_tool.network.apply_pruning.links = "archive-writer"

        config.flf_lattice_tool.network.archive_writer.type = "archive-writer"
        config.flf_lattice_tool.network.archive_writer.path = (
            "pruned_lattice.cache.$(TASK)"
        )
        config.flf_lattice_tool.network.archive_writer.format = output_format
        config.flf_lattice_tool.network.archive_writer.flf.partial.keys = "am lm"
        config.flf_lattice_tool.network.archive_writer.flf.partial.add = False
        config.flf_lattice_tool.network.archive_writer.links = "sink"
        if output_format == "lattice-processor":
            config.flf_lattice_tool.network.archive_writer.lattice_processor.pronunciation_scale = (
                pronunciation_scale
            )

        config.flf_lattice_tool.network.sink.type = "sink"
        config.flf_lattice_tool.network.sink.warn_on_empty_lattice = True
        config.flf_lattice_tool.network.sink.error_on_empty_lattice = False

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        return super().hash({"config": config, "exe": kwargs["crp"].flf_tool_exe})
