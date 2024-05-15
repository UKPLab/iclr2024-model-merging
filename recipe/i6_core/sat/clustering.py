__all__ = ["BayesianInformationClusteringJob"]

import xml.etree.cElementTree as etree
import collections
import os

from sisyphus import *

Path = setup_path(__package__)

import i6_core.rasr as rasr
from i6_core.util import MultiOutputPath
from .flow import segment_clustering_flow


class BayesianInformationClusteringJob(rasr.RasrCommand, Job):
    """
    Generate a coprus-key-map based on the Bayesian information criterion. Each concurrent is clustered independently.
    """

    def __init__(
        self,
        crp,
        feature_flow,
        minframes=0,
        mincluster=2,
        maxcluster=100000,
        threshold=0,
        _lambda=1,
        minalpha=0.40,
        maxalpha=0.60,
        alpha=-1,
        amalgamation=0,
        infile=None,
        job_name="bic_cluster",
        rtf=0.02,
        mem=None,
        extra_config=None,
        extra_post_config=None,
        **kwargs,
    ):
        self.set_vis_name("Extract %s" % job_name)

        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = BayesianInformationClusteringJob.create_config(
            **kwargs
        )
        self.concurrent = crp.concurrent
        self.exe = (
            crp.feature_extraction_exe
            if crp.feature_extraction_exe is not None
            else self.default_exe("feature-extraction")
        )
        self.cluster_flow = segment_clustering_flow(
            file="cluster.map.$(TASK)", **kwargs
        )

        self.log_file = self.log_file_output_path("cluster", crp, True)
        self.num_speakers = self.output_var("num_speakers", True)
        self.segment_dir = self.output_path("segments", True)
        self.segment_path = MultiOutputPath(
            self, "segments/speaker.$(TASK)", self.segment_dir
        )
        self.speaker_map_file = self.output_path("speaker.map")
        self.cluster_map_file = self.output_path("cluster.map.xml")

        self.rqmt = {
            "time": 24,
            "cpu": 1,
            "mem": min(max(rtf / self.concurrent * crp.corpus_duration, 1), 16),
        }
        if mem:
            self.rqmt["mem"] = mem

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task(
            "run", resume="run", rqmt=self.rqmt, args=range(1, self.concurrent + 1)
        )
        yield Task("merge", resume="merge", mini_task=True)

    def run(self, task_id):
        self.run_script(task_id, self.log_file[task_id])

    def merge(self):
        speaker_map = collections.defaultdict(list)

        for i in range(1, self.concurrent + 1):
            with open("cluster.map.{}".format(i)) as f:
                doc = etree.parse(f)
                for item in doc.findall("map-item"):
                    speaker = "{}_{}".format(item.attrib["value"], i)
                    speaker_map[speaker].append(item.attrib["key"])

        self.num_speakers.set(len(speaker_map))

        with open(self.speaker_map_file.get_path(), "wt") as smf:
            with open(self.cluster_map_file.get_path(), "wt") as cmf:
                cmf.write('<?xml version="1.0" encoding="utf-8" ?>\n')
                cmf.write("<coprus-key-map>\n")  # misspelled on purpose
                for idx, speaker in enumerate(sorted(speaker_map), 1):
                    smf.write("%s\n" % speaker)
                    with open(
                        os.path.join(self.segment_dir.get_path(), "speaker.%d" % idx),
                        "wt",
                    ) as ssf:
                        for segment in speaker_map[speaker]:
                            ssf.write("%s\n" % segment)
                            cmf.write(
                                '  <map-item key="%s" value="cluster.%d"/>\n'
                                % (segment, idx)
                            )
                cmf.write("</coprus-key-map>")  # misspelled on purpose

    def create_files(self):
        self.write_config(self.config, self.post_config, "clustering.config")
        self.cluster_flow.write_to_file("cluster.flow")
        self.write_run_script(self.exe, "clustering.config")

    @classmethod
    def create_config(
        cls, crp, feature_flow, extra_config, extra_post_config, **kwargs
    ):
        config, post_config = rasr.build_config_from_mapping(
            crp, {"corpus": "extraction.corpus"}, parallelize=True
        )
        config.extraction.feature_extraction.file = "cluster.flow"

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def create_flow(cls, feature_flow, **kwargs):
        net = segment_clustering_flow(feature_flow, **kwargs)
        return net

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        cluster_flow = cls.create_flow(**kwargs)
        return (
            kwargs["job_name"]
            + "."
            + super().hash(
                {
                    "config": config,
                    "flow": cluster_flow,
                    "exe": kwargs["crp"].feature_extraction_exe,
                }
            )
        )
