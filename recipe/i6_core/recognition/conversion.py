__all__ = ["LatticeToCtmJob"]

import shutil

from sisyphus import *

Path = setup_path(__package__)

import i6_core.rasr as rasr
import i6_core.util as util


class LatticeToCtmJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        lattice_cache,
        *,
        parallelize=False,
        encoding="utf-8",
        fill_empty_segments=False,
        best_path_algo="bellman-ford",
        extra_config=None,
        extra_post_config=None,
    ):
        self.set_vis_name("Convert Lattice to CTM file")

        kwargs = locals()
        del kwargs["self"]

        self.best_path_algo = best_path_algo
        self.config, self.post_config = LatticeToCtmJob.create_config(**kwargs)
        self.concurrent = crp.concurrent if parallelize else 1
        self.exe = self.select_exe(crp.flf_tool_exe, "flf-tool")
        self.lattice_cache = lattice_cache

        self.out_log_file = self.log_file_output_path(
            "lattice_to_ctm", crp, self.concurrent > 1
        )
        self.out_ctm_file = self.output_path("lattice.ctm")

        self.rqmt = {
            "time": max(crp.corpus_duration / (5.0 * self.concurrent), 0.5),
            "cpu": 1,
            "mem": 4,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task(
            "run", resume="run", rqmt=self.rqmt, args=range(1, self.concurrent + 1)
        )
        if self.concurrent > 1:
            yield Task("merge", mini_task=True)

    def create_files(self):
        self.write_config(self.config, self.post_config, "lattice_to_ctm.config")
        self.write_run_script(self.exe, "lattice_to_ctm.config")

    def run(self, task_id):
        log_file = (
            self.out_log_file if self.concurrent <= 1 else self.out_log_file[task_id]
        )
        self.run_script(task_id, log_file)
        if self.concurrent <= 1:
            shutil.move("lattice.ctm.1", self.out_ctm_file.get_path())

    def merge(self):
        with open(self.out_ctm_file.get_path(), "wt") as out:
            for t in range(1, self.concurrent + 1):
                with open("lattice.ctm.%d" % t, "rt") as ctm:
                    shutil.copyfileobj(ctm, out)

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("lattice_to_ctm.log")

    @classmethod
    def create_config(
        cls,
        crp,
        lattice_cache,
        parallelize,
        encoding,
        fill_empty_segments,
        best_path_algo,
        extra_config,
        extra_post_config,
        **kwargs,
    ):
        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "corpus": "flf-lattice-tool.corpus",
                "lexicon": "flf-lattice-tool.lexicon",
            },
            parallelize=parallelize,
        )
        # segment
        config.flf_lattice_tool.network.initial_nodes = "segment"
        config.flf_lattice_tool.network.segment.type = "speech-segment"
        config.flf_lattice_tool.network.segment.links = (
            "0->archive-reader:1 0->dump-ctm:1"
        )

        # read lattice
        config.flf_lattice_tool.network.archive_reader.type = "archive-reader"
        config.flf_lattice_tool.network.archive_reader.links = "to-lemma"
        config.flf_lattice_tool.network.archive_reader.format = "flf"
        config.flf_lattice_tool.network.archive_reader.path = lattice_cache
        config.flf_lattice_tool.network.archive_reader.flf.append.keys = "confidence"
        config.flf_lattice_tool.network.archive_reader.flf.append.confidence.scale = 0.0

        # store ctms with (raw) confidences
        config.flf_lattice_tool.network.to_lemma.type = "map-alphabet"
        config.flf_lattice_tool.network.to_lemma.links = "add-word-confidence"
        config.flf_lattice_tool.network.to_lemma.map_input = "to-lemma"
        config.flf_lattice_tool.network.to_lemma.project_input = True

        config.flf_lattice_tool.network.add_word_confidence.type = "fCN-features"
        config.flf_lattice_tool.network.add_word_confidence.links = "best"
        config.flf_lattice_tool.network.add_word_confidence.rescore_mode = (
            "in-place-cached"
        )
        config.flf_lattice_tool.network.add_word_confidence.confidence_key = (
            "confidence"
        )

        config.flf_lattice_tool.network.best.type = "best"
        config.flf_lattice_tool.network.best.links = "dump-ctm"
        if best_path_algo:
            config.flf_lattice_tool.network.best.algorithm = best_path_algo

        config.flf_lattice_tool.network.dump_ctm.type = "dump-traceback"
        config.flf_lattice_tool.network.dump_ctm.links = "sink:1"
        config.flf_lattice_tool.network.dump_ctm.format = "ctm"
        config.flf_lattice_tool.network.dump_ctm.dump.channel = "lattice.ctm.$(TASK)"
        config.flf_lattice_tool.network.dump_ctm.ctm.scores = "confidence"
        if fill_empty_segments:
            config.flf_lattice_tool.network.dump_ctm.ctm.fill_empty_segments = True

        # sink
        config.flf_lattice_tool.network.sink.type = "sink"
        config.flf_lattice_tool.network.sink.warn_on_empty_lattice = True
        config.flf_lattice_tool.network.sink.error_on_empty_lattice = False

        post_config.channels["$(TASK)"].encoding = encoding

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        return super().hash({"config": config, "exe": kwargs["crp"].flf_tool_exe})
