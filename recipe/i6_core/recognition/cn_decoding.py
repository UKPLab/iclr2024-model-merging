__all__ = ["CNDecodingJob"]

from sisyphus import *

Path = setup_path(__package__)

import shutil

import i6_core.rasr as rasr
import i6_core.util as util


class CNDecodingJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        lattice_path,
        lm_scale,
        pron_scale=1.0,
        write_cn=False,
        extra_config=None,
        extra_post_config=None,
    ):
        self.set_vis_name("CN decoding")

        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = self.create_config(**kwargs)
        self.exe = self.select_exe(crp.flf_tool_exe, "flf-tool")
        self.concurrent = crp.concurrent
        self.write_cn = write_cn

        self.out_log_file = self.log_file_output_path("cn_decoding", crp, True)
        self.out_single_lattice_caches = dict(
            (
                task_id,
                self.output_path("confusion_lattice.cache.%d" % task_id, cached=True),
            )
            for task_id in range(1, crp.concurrent + 1)
        )
        self.out_ctm_file = self.output_path("lattice.ctm")
        if self.write_cn:
            self.out_lattice_bundle = self.output_path(
                "confusion_lattice.bundle", cached=True
            )
            self.out_lattice_path = util.MultiOutputPath(
                self,
                "confusion_lattice.cache.$(TASK)",
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
        yield Task("merge", mini_task=True)

    def create_files(self):
        self.write_config(self.config, self.post_config, "cn_decoding.config")
        self.write_run_script(self.exe, "cn_decoding.config")
        if self.write_cn:
            util.write_paths_to_file(
                self.out_lattice_bundle, self.out_single_lattice_caches.values()
            )

    def run(self, task_id):
        self.run_script(task_id, self.out_log_file[task_id])
        if self.write_cn:
            shutil.move(
                "confusion_lattice.cache.%d" % task_id,
                self.out_single_lattice_caches[task_id].get_path(),
            )

    def merge(self):
        with open(self.out_ctm_file.get_path(), "wt") as out:
            for t in range(1, self.concurrent + 1):
                with open("lattice.ctm.%d" % t, "rt") as ctm:
                    shutil.copyfileobj(ctm, out)

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("cn_decoding.log.%d" % task_id)
        util.delete_if_exists("confusion_lattice.cache.%d" % task_id)

    @classmethod
    def create_config(
        cls,
        crp,
        lattice_path,
        lm_scale,
        pron_scale,
        write_cn,
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
            "0->archive-reader:1 0->dump-ctm:1"
        )
        if write_cn:
            config.flf_lattice_tool.network.segment.links += " 0->archive-writer:1"

        config.flf_lattice_tool.network.archive_reader.type = "archive-reader"
        config.flf_lattice_tool.network.archive_reader.format = "flf"
        config.flf_lattice_tool.network.archive_reader.path = lattice_path
        config.flf_lattice_tool.network.archive_reader.flf.append.keys = "confidence"
        config.flf_lattice_tool.network.archive_reader.flf.append.confidence.scale = 0.0
        config.flf_lattice_tool.network.archive_reader.links = "0->rescale:0"

        config.flf_lattice_tool.network.rescale.type = "rescale"
        config.flf_lattice_tool.network.rescale.lm.scale = lm_scale
        config.flf_lattice_tool.network.rescale.am.scale = 1.0
        config.flf_lattice_tool.network.rescale.links = "scale-pronunciation"

        config.flf_lattice_tool.network.scale_pronunciation.type = (
            "extend-by-pronunciation-score"
        )
        config.flf_lattice_tool.network.scale_pronunciation.key = "am"
        config.flf_lattice_tool.network.scale_pronunciation.scale = pron_scale
        config.flf_lattice_tool.network.scale_pronunciation.rescore_mode = (
            "in-place-cached"
        )
        config.flf_lattice_tool.network.scale_pronunciation.links = "to-lemma"

        config.flf_lattice_tool.network.to_lemma.type = "map-alphabet"
        config.flf_lattice_tool.network.to_lemma.map_input = "to-lemma"
        config.flf_lattice_tool.network.to_lemma.project_input = True
        config.flf_lattice_tool.network.to_lemma.links = "apply-pre-processing"

        # not sure why this is done, found it in kazukis setup
        config.flf_lattice_tool.network.apply_pre_processing.type = "copy"
        config.flf_lattice_tool.network.apply_pre_processing.links = "cn-builder"

        config.flf_lattice_tool.network.cn_builder.type = "pivot-arc-CN-builder"
        config.flf_lattice_tool.network.cn_builder.confidence_key = "confidence"
        config.flf_lattice_tool.network.cn_builder.map = False
        config.flf_lattice_tool.network.cn_builder.distance = "weighted-pivot-time"
        config.flf_lattice_tool.network.cn_builder.weighted_pivot_time.posterior_impact = (
            0.1
        )
        config.flf_lattice_tool.network.cn_builder.weighted_pivot_time.edit_distance = (
            False
        )
        config.flf_lattice_tool.network.cn_builder.weighted_pivot_time.fast = False
        config.flf_lattice_tool.network.cn_builder.links = "0->dump-ctm:0"

        if write_cn:
            config.flf_lattice_tool.network.cn_builder.links += "1->cn-archive-writer:0"

            config.flf_lattice_tool.network.cn_archive_writer.type = "CN-archive-writer"
            config.flf_lattice_tool.network.cn_archive_writer.path = "cn.cache.$(TASK)"
            config.flf_lattice_tool.network.cn_archive_writer.format = "xml"
            config.flf_lattice_tool.network.cn_archive_writer.links = "cn-sink"

            config.flf_lattice_tool.network.cn_sink.type = "sink"
            config.flf_lattice_tool.network.cn_sink.sink_type = "CN"
            config.flf_lattice_tool.network.cn_sink.warn_on_empty = True
            config.flf_lattice_tool.network.cn_sink.error_on_empty = False

        config.flf_lattice_tool.network.dump_ctm.type = "dump-traceback"
        config.flf_lattice_tool.network.dump_ctm.format = "ctm"
        config.flf_lattice_tool.network.dump_ctm.ctm.scores = "confidence"
        config.flf_lattice_tool.network.dump_ctm.dump.channel = "lattice.ctm.$(TASK)"
        config.flf_lattice_tool.network.dump_ctm.links = "sink"

        config.flf_lattice_tool.network.sink.type = "sink"
        config.flf_lattice_tool.network.sink.warn_on_empty = True
        config.flf_lattice_tool.network.sink.error_on_empty = False

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        return super().hash({"config": config, "exe": kwargs["crp"].flf_tool_exe})
