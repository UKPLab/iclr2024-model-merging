__all__ = [
    "MapSegmentsWithBundlesJob",
    "RemapSegmentsWithBundlesJob",
    "ClusterMapToSegmentListJob",
    "RemapSegmentsJob",
]

import collections
import logging
import xml.etree.ElementTree as ET

from sisyphus import *

Path = setup_path(__package__)

from i6_core.util import *


class MapSegmentsWithBundlesJob(Job):
    def __init__(self, old_segments, cluster_map, files, filename="cluster.$(TASK)"):
        self.old_segments = old_segments
        self.cluster_map = cluster_map
        self.files = files
        self.filename = filename

        self.out_bundle_dir = self.output_path("bundles", True)
        self.out_bundle_path = MultiOutputPath(
            self,
            os.path.join("bundles", "%s") % self.filename,
            self.out_bundle_dir,
            cached=True,
        )

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        segment_map = {}
        for idx, seg in self.old_segments.items():
            p = tk.uncached_path(seg)
            with open(p, "rt") as f:
                for line in f:
                    segment_map[line.strip()] = idx

        t = ET.parse(tk.uncached_path(self.cluster_map))
        cluster_map = collections.defaultdict(list)
        for mit in t.findall("map-item"):
            cluster_map[mit.attrib["value"]].append(
                mit.attrib["key"]
            )  # use full segment name, as the name of the segment is not unique inside some corpora

        for cluster, segments in cluster_map.items():
            seg_files = set()
            for segment in segments:
                if segment in segment_map:
                    seg_file = tk.uncached_path(self.files[segment_map[segment]])
                    seg_files.add(seg_file)

            with open(
                os.path.join(self.out_bundle_dir.get_path(), cluster + ".bundle"), "wt"
            ) as f:
                for seg_file in seg_files:
                    f.write("%s\n" % seg_file)


class RemapSegmentsWithBundlesJob(Job):
    def __init__(self, old_segments, new_segments, files):
        self.old_segments = old_segments
        self.new_segments = new_segments
        self.files = files

        self.bundle_dir = self.output_path("bundles", True)
        self.bundle_path = MultiOutputPath(
            self,
            os.path.join("bundles", "remapped.$(TASK).bundle"),
            self.bundle_dir,
            cached=True,
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        old_segment_map = {}
        for idx, seg in self.old_segments.items():
            p = tk.uncached_path(seg)
            with open(p, "rt") as f:
                for line in f:
                    old_segment_map[line.strip()] = idx

        for (
            idx,
            seg,
        ) in self.new_segments.items():
            old_idxs = set()
            p = tk.uncached_path(seg)
            with open(p, "rt") as f:
                for line in f:
                    line = line.strip()
                    try:
                        old_idx = old_segment_map[line]
                    except KeyError:
                        # sometimes the new index list is the full segment name, but the old one is only the segment name itself
                        old_idx = old_segment_map[line.split("/")[-1]]
                    old_idxs.add(old_idx)
            with open(
                os.path.join(self.bundle_dir.get_path(), "remapped.%d.bundle" % idx),
                "wt",
            ) as out:
                for old_idx in old_idxs:
                    out.write(tk.uncached_path(self.files[old_idx]))
                    out.write("\n")


class ClusterMapToSegmentListJob(Job):
    """
    Creates segment files in relation to a speaker cluster map

    WARNING: This job has broken (non-portable) hashes and is not really useful anyway,
             please use this only for existing pipelines
    """

    def __init__(self, cluster_map, filename="cluster.$(TASK)"):
        self.cluster_map = cluster_map

        self.out_segment_dir = self.output_path("segments", True)
        self.out_segment_path = MultiOutputPath(
            self,
            os.path.join(self.out_segment_dir.get_path(), filename),
            self.out_segment_dir,
            cached=True,
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        t = ET.parse(tk.uncached_path(self.cluster_map))
        cluster_map = collections.defaultdict(list)
        for mit in t.findall("map-item"):
            cluster_map[mit.attrib["value"]].append(mit.attrib["key"])

        for cluster, segments in cluster_map.items():
            with open(
                os.path.join(self.out_segment_dir.get_path(), cluster), "wt"
            ) as f:
                for seg in segments:
                    f.write("%s\n" % seg)


class RemapSegmentsJob(Job):
    def __init__(self, old_segments, new_segments, cache_paths):
        assert len(old_segments) == len(cache_paths)
        self.old_segments = old_segments
        self.new_segments = new_segments
        self.cache_paths = cache_paths

        self.bundle_dir = self.output_path("bundles", True)
        self.bundle_path = MultiOutputPath(
            self,
            os.path.join("bundles", "feature.$(TASK).bundle"),
            self.bundle_dir,
            cached=True,
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        segment_map = {}
        for i, p in enumerate(self.old_segments):
            for line in open(tk.uncached_path(p), "rt"):
                line = line.strip()
                if len(line) > 0:
                    segment_map[line] = i

        bundle_map = [set() for i in range(len(self.new_segments))]
        for i, p in enumerate(self.new_segments):
            for line in open(tk.uncached_path(p), "rt"):
                line = line.strip()
                if len(line) > 0:
                    bundle_map[i].add(segment_map[line])

        for i, bs in enumerate(bundle_map):
            with open(
                os.path.join(self.bundle_dir.get_path(), "feature.%d.bundle" % (i + 1)),
                "wt",
            ) as f:
                for b in bs:
                    f.write("%s\n" % tk.uncached_path(self.cache_paths[b]))
