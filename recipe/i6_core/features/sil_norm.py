__all__ = [
    "samples_with_silence_normalization_flow",
    "ExtractSilenceNormalizationMapJob",
    "ExtractSegmentSilenceNormalizationMapJob",
    "UnwarpTimesInCTMJob",
]

from sisyphus import *

Path = setup_path(__package__)

import gzip
import xml.etree.ElementTree as ET

import i6_core.rasr as rasr


def samples_with_silence_normalization_flow(
    audio_format="wav", dc_detection=True, dc_params=None, silence_params=None
):
    _dc_params = {
        "min-dc-length": 0.01,
        "max-dc-increment": 0.9,
        "min-non-dc-segment-length": 0.021,
    }
    _silence_params = {
        "absolute-silence-threshold": 250,
        "discard-unsure-segments": True,
        "min-surrounding-silence": 0.1,
        "fill-up-silence": True,
        "silence-ratio": 0.25,
        "silence-threshold": 0.05,
    }
    if dc_params is not None:
        _dc_params.update(dc_params)
    if silence_params is not None:
        _silence_params.update(silence_params)

    net = rasr.FlowNetwork()

    net.add_output("samples")
    net.add_param(["input-file", "start-time", "end-time", "track"])

    samples = net.add_node(
        "audio-input-file-" + audio_format,
        "samples",
        {
            "file": "$(input-file)",
            "start-time": "$(start-time)",
            "end-time": "$(end-time)",
        },
    )

    demultiplex = net.add_node(
        "generic-vector-s16-demultiplex", "demultiplex", track="$(track)"
    )
    net.link(samples, demultiplex)

    convert = net.add_node("generic-convert-vector-s16-to-vector-f32", "convert")
    net.link(demultiplex, convert)

    sil_norm = net.add_node("signal-silence-normalization", "silence-normalization")
    net.link(convert, sil_norm)
    warp_time = net.add_node("warp-time", "warp-time", {"start-time": "$(start-time)"})
    if dc_detection:
        dc_detection = net.add_node("signal-dc-detection", "dc-detection", _dc_params)
        net.link(sil_norm, dc_detection)
        net.link(dc_detection, warp_time)
    else:
        net.link(sil_norm, warp_time)

    net.link(warp_time, "network:samples")

    net.config = rasr.RasrConfig()
    for k, v in _silence_params:
        net.config[sil_norm][k] = v

    return net


class ExtractSilenceNormalizationMapJob(Job):
    def __init__(self, log_files):
        self.log_files = log_files

        self.sil_norm_map = self.output_path("silence-normalization.map")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        normsil_label = "silence-normalization warping map:"
        warping_label = "warping map:"

        had_normsil_mal = False
        had_warping_map = False

        maps = {}

        for log in self.log_files:
            path = tk.uncached_path(log)
            open_fun = gzip.open if path.endswith(".gz") else open
            with open_fun(path, "rt") as f:
                tree = ET.parse(f)
            for rec in tree.findall("*/recording"):
                recname = rec.get("name")
                if recname not in maps:
                    maps[recname] = set()
                for seg in rec.findall("segment"):
                    for info in seg.findall("information"):
                        comp = info.get("component")
                        if comp.endswith(
                            ".silence-normalization"
                        ) and info.text.strip().startswith(normsil_label):
                            assert not had_warping_map
                            had_normsil_mal = True
                            for item in info.text.strip()[len(normsil_label) :].split():
                                maps[recname].add(tuple(map(float, item.split(":"))))
                        if comp.endswith(".warp-time") and info.text.strip().startswith(
                            warping_label
                        ):
                            assert not had_normsil_mal
                            had_warping_map = True
                            for item in info.text.strip()[len(warping_label) :].split():
                                maps[recname].add(tuple(map(float, item.split(":"))))

        with open(self.sil_norm_map.get_path(), "wt") as f:
            for recname in maps.keys():
                l = list(maps[recname])
                l.sort(key=lambda tup: tup[0])
                l = [recname] + ["%.2f:%.2f" % (e[0], e[1]) for e in l]
                f.write(" ".join(l) + "\n")


class ExtractSegmentSilenceNormalizationMapJob(Job):
    def __init__(self, log_files):
        self.log_files = log_files

        self.sil_norm_map = self.output_path("segment.silence-normalization.map")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        normsil_label = "silence-normalization warping map:"
        warping_label = "warping map:"

        had_normsil_mal = False
        had_warping_map = False

        maps = {}

        for log in self.log_files:
            path = tk.uncached_path(log)
            open_fun = gzip.open if path.endswith(".gz") else open
            with open_fun(path, "rt") as f:
                tree = ET.parse(f)

            for rec in tree.findall("*/recording"):
                for seg in rec.findall("segment"):
                    seg_full_name = seg.get("full-name")
                    if seg_full_name not in maps:
                        maps[seg_full_name] = set()
                    for info in seg.findall("information"):
                        comp = info.get("component")
                        if comp.endswith(
                            ".silence-normalization"
                        ) and info.text.strip().startswith(normsil_label):
                            assert not had_warping_map
                            had_normsil_mal = True
                            for item in info.text.strip()[len(normsil_label) :].split():
                                maps[seg_full_name].add(
                                    tuple(map(float, item.split(":")))
                                )
                        if comp.endswith(".warp-time") and info.text.strip().startswith(
                            warping_label
                        ):
                            assert not had_normsil_mal
                            had_warping_map = True
                            for item in info.text.strip()[len(warping_label) :].split():
                                maps[seg_full_name].add(
                                    tuple(map(float, item.split(":")))
                                )

        with open(self.sil_norm_map.get_path(), "wt") as f:
            for segname in maps.keys():
                l = list(maps[segname])
                l.sort(key=lambda tup: tup[0])
                l = [segname] + ["%.2f:%.2f" % (e[0], e[1]) for e in l]
                f.write(" ".join(l) + "\n")


class UnwarpTimesInCTMJob(Job):
    def __init__(self, ctm_file, sil_norm_map):
        self.ctm_file = ctm_file
        self.sil_norm_map = sil_norm_map

        self.unwarped_ctm = self.output_path("unwarped.ctm")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        silence_map = {}
        with open(tk.uncached_path(self.sil_norm_map), "rt") as f:
            for line in f:
                items = line.strip().split()
                if len(items) > 0:
                    silence_map[items[0]] = [
                        (float(a), float(b))
                        for item in items[1:]
                        for a, b in [item.split(":")]
                    ]

        def apply(recname, time):
            time_offset = 0
            for (a, b) in silence_map[recname]:
                if time >= a:
                    time_offset = b - a
                    assert time_offset >= 0
                else:
                    break
            return time + time_offset

        with open(tk.uncached_path(self.ctm_file), "rt") as infile:
            with open(self.unwarped_ctm.get_path(), "wt") as outfile:
                for line in infile:
                    items = line.strip().split()
                    if len(items) < 3 or line.startswith(";"):
                        outfile.write(line)
                        continue
                    outfile.write(
                        "%s %s %.3f %s\n"
                        % (
                            items[0],
                            items[1],
                            apply(items[0], float(items[2])),
                            " ".join(items[3:]),
                        )
                    )
