__all__ = ["ToneJob"]

import concurrent.futures as futures
import gzip
import os
import random
import shutil
import subprocess as sp
import wave

from sisyphus import *

Path = setup_path(__package__)

from .common import samples_flow as default_samples_flow
import i6_core.rasr as rasr
import i6_core.util as util


class ToneJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        timestamp_flow,
        *,
        samples_flow=None,
        min_length=0.5,
        timestamp_port="features",
        extract_concurrent=4,
        rtf=0.1,
        mem=2.0,
        extra_dump_config=None,
        extra_dump_post_config=None,
        extra_convert_config=None,
        extra_convert_post_config=None,
    ):
        kwargs = locals()
        del kwargs["self"]

        self.min_length = min_length
        self.extract_concurrent = extract_concurrent
        self.dump_config, self.dump_post_config = self.create_dump_config(**kwargs)
        self.dump_flow = self.create_dump_flow(**kwargs)
        self.convert_config, self.convert_post_config = self.create_convert_config(
            **kwargs
        )
        self.convert_flow = self.create_convert_flow(**kwargs)
        self.exe = (
            crp.feature_extraction_exe
            if crp.feature_extraction_exe is not None
            else self.default_exe("feature-extraction")
        )
        self.concurrent = crp.concurrent

        self.out_dump_log_file = self.log_file_output_path("dump", crp, True)
        self.out_convert_log_file = self.log_file_output_path("convert", crp, True)
        self.out_single_feature_caches = dict(
            (task_id, self.output_path("tone.cache.%d" % task_id, cached=True))
            for task_id in range(1, crp.concurrent + 1)
        )
        self.out_feature_bundle = self.output_path("tone.cache.bundle", cached=True)
        self.out_feature_path = util.MultiOutputPath(
            self, "tone.cache.$(TASK)", self.out_single_feature_caches, cached=True
        )

        self.dump_rqmt = {
            "time": max(crp.corpus_duration * rtf / crp.concurrent, 0.5),
            "cpu": 1,
            "mem": mem,
        }
        self.extract_pitch_rqmt = {
            "time": max(crp.corpus_duration * rtf / self.extract_concurrent, 0.5),
            "cpu": extract_concurrent,
            "mem": mem,
        }
        self.convert_rqmt = {
            "time": max(crp.corpus_duration * rtf / crp.concurrent, 0.5),
            "cpu": 1,
            "mem": mem,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task(
            "dump",
            resume="dump",
            rqmt=self.dump_rqmt,
            args=range(1, self.concurrent + 1),
        )
        yield Task(
            "extract_pitch", resume="extract_pitch", rqmt=self.extract_pitch_rqmt
        )
        yield Task(
            "convert",
            resume="convert",
            rqmt=self.convert_rqmt,
            args=range(1, self.concurrent + 1),
        )

    def create_files(self):
        self.write_config(self.dump_config, self.dump_post_config, "dump.config")
        self.dump_flow.write_to_file("dump.flow")
        self.write_run_script(self.exe, "dump.config", filename="dump.sh")
        os.mkdir("dump")

        self.write_config(
            self.convert_config, self.convert_post_config, "convert.config"
        )
        self.convert_flow.write_to_file("convert.flow")
        self.write_run_script(self.exe, "convert.config", filename="convert.sh")
        util.write_paths_to_file(
            self.out_feature_bundle, self.out_single_feature_caches.values()
        )

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        if cmd == "./dump.sh":
            util.backup_if_exists("dump.log.%d" % task_id)
        elif cmd == "./convert.sh":
            util.backup_if_exists("convert.log.%d" % task_id)
            util.delete_if_zero("tone.cache.%d" % task_id)

    def dump(self, task_id):
        self.run_script(task_id, self.out_dump_log_file[task_id], cmd="./dump.sh")

    def extract_pitch(self):
        lib = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../lib/pitch_extraction/")
        )
        getf0 = os.path.join(lib, "getf0.py")
        getpitch = os.path.join(lib, "getpitch.m")

        wav_files = []
        for root, dirs, files in os.walk("dump"):
            for filename in files:
                if filename.endswith(".wav"):
                    abspath = os.path.join(root, filename)
                    wav_files.append(abspath)

        def process_file(abspath):
            w = wave.open(abspath, "r")
            wav_length = w.getnframes() / float(w.getframerate())
            random.seed(w.getnframes())  # set the seed to wave length
            del w

            print("creating pitch for", abspath, "with length", wav_length)
            pitch = None

            if wav_length >= self.min_length:
                f0_path = abspath[:-3] + "f0.txt"
                try:
                    sp.check_call(["xvfb-run", "-a", getf0, "-o", f0_path, abspath])
                    output = sp.check_output([getpitch, f0_path])
                    pitch = [
                        float(l.strip())
                        for l in output.decode("utf8").split("\n")
                        if len(l.strip()) > 0
                    ]
                except sp.CalledProcessError:
                    pass

            if wav_length < self.min_length or pitch is None:
                # The pitch extraction utilities are not robust enough to deal with very short recordings
                if wav_length < self.min_length:
                    print("segment is too short, using random pitch")
                else:
                    print("using random pitch because pitch-extraction failed")
                pitch = [random.random() for i in range(int(wav_length * 100))]

            with gzip.open(abspath[:-3] + "xml.gz", "wt") as out:
                out.write('<?xml version="1.0" encoding="utf8"?>\n<sprint>\n')
                for i, p in enumerate(pitch):
                    out.write(
                        '<dump-data node="dummy"><vector-f32 start="%.4f" end="%.4f" size="1">%.10f</vector-f32></dump-data>\n'
                        % (i * 0.01, (i + 1) * 0.01, p)
                    )
                out.write("</sprint>")

        with futures.ThreadPoolExecutor(max_workers=self.extract_concurrent) as e:
            e.map(process_file, wav_files)

    def convert(self, task_id):
        self.run_script(task_id, self.out_convert_log_file[task_id], cmd="./convert.sh")
        shutil.move(
            "tone.cache.%d" % task_id,
            self.out_single_feature_caches[task_id].get_path(),
        )

    @classmethod
    def create_dump_flow(cls, crp, samples_flow, **kwargs):
        if samples_flow is None:
            samples_flow = default_samples_flow(crp.audio_format)

        net = rasr.FlowNetwork()
        net.add_param("id")
        net.add_output("features")

        samples_mapping = net.add_net(samples_flow)
        net.interconnect_inputs(samples_flow, samples_mapping)

        samples = samples_mapping[samples_flow.get_output_links("samples").pop()]

        convert = net.add_node(
            "generic-convert-vector-f32-to-vector-s16", "convert-back"
        )
        net.link(samples, convert)

        write = net.add_node(
            "audio-output-file-wav", "write", {"file": "dump/$(id).wav"}
        )
        net.link(convert, write)

        convert2 = net.add_node(
            "generic-convert-vector-s16-to-vector-f32", "convert-again"
        )
        net.link(write, convert2)
        net.link(convert2, "network:features")

        return net

    @classmethod
    def create_dump_config(
        cls, crp, samples_flow, extra_dump_config, extra_dump_post_config, **kwargs
    ):
        dump_flow = cls.create_dump_flow(crp, samples_flow)

        config, post_config = rasr.build_config_from_mapping(
            crp, {"corpus": "extraction.corpus"}, parallelize=True
        )
        config.extraction.feature_extraction.file = "dump.flow"
        config.extraction.feature_extraction["*"].allow_overwrite = True

        dump_flow.apply_config("extraction.feature-extraction", config, post_config)

        config._update(extra_dump_config)
        post_config._update(extra_dump_post_config)

        return config, post_config

    @classmethod
    def create_convert_flow(cls, crp, timestamp_flow, timestamp_port, **kwargs):
        net = rasr.FlowNetwork()
        net.add_param("id")
        net.add_param("start-time")
        net.add_output("features")

        text_input = net.add_node(
            "generic-vector-f32-text-input",
            "reader",
            {"offset": "$(start-time)", "file": "dump/$(id).xml.gz"},
        )

        timestamp_mapping = net.add_net(timestamp_flow)
        timestamp = timestamp_mapping[
            timestamp_flow.get_output_links(timestamp_port).pop()
        ]

        sync = net.add_node(
            "timestamp-copy", "synchronization", {"ignore-errors": True}
        )
        net.link(timestamp, sync + ":target")
        net.link(text_input, sync)

        norm = net.add_node(
            "signal-normalization",
            "normalization",
            {"type": "mean-and-variance", "length": "infinite", "right": "infinite"},
        )
        net.link(sync, norm)

        repeat = net.add_node("signal-repeating-frame-prediction", "feature-sync")
        net.link(timestamp, repeat + ":target")
        net.link(norm, repeat)

        cache = net.add_node(
            "generic-cache", "out-cache", {"path": "tone.cache.$(TASK)", "id": "$(id)"}
        )
        net.link(repeat, cache)
        net.link(cache, "network:features")

        return net

    @classmethod
    def create_convert_config(
        cls,
        crp,
        timestamp_flow,
        timestamp_port,
        extra_convert_config,
        extra_convert_post_config,
        **kwargs,
    ):
        convert_flow = cls.create_convert_flow(crp, timestamp_flow, timestamp_port)

        config, post_config = rasr.build_config_from_mapping(
            crp, {"corpus": "extraction.corpus"}, parallelize=True
        )
        config.extraction.feature_extraction.file = "convert.flow"
        config.extraction.feature_extraction["*"].allow_overwrite = True

        convert_flow.apply_config("extraction.feature-extraction", config, post_config)

        config._update(extra_convert_config)
        post_config._update(extra_convert_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        dump_config, dump_post_config = cls.create_dump_config(**kwargs)
        dump_flow = cls.create_dump_flow(**kwargs)
        convert_config, convert_post_config = cls.create_convert_config(**kwargs)
        convert_flow = cls.create_convert_flow(**kwargs)
        return super().hash(
            {
                "dump_config": dump_config,
                "dump_flow": dump_flow,
                "convert_config": convert_config,
                "convert_flow": convert_flow,
                "exe": kwargs["crp"].feature_extraction_exe,
            }
        )
