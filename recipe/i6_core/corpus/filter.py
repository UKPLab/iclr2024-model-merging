__all__ = [
    "FilterSegmentsByAlignmentConfidenceJob",
    "FilterCorpusBySegmentsJob",
    "FilterSegmentsByListJob",
    "FilterCorpusRemoveUnknownWordSegmentsJob",
    "FilterCorpusBySegmentDurationJob",
]

import gzip
import logging
import xml.etree.cElementTree as ET

import numpy as np

from i6_core.util import MultiOutputPath

from i6_core.lib import corpus
from i6_core.util import chunks, uopen

from sisyphus import *

Path = setup_path(__package__)


class FilterSegmentsByListJob(Job):
    def __init__(self, segment_files, filter_list, invert_match=False):
        """
        Filters segment list file using a given list of segments, which is either used as black or as white list
        :param dict[int,Path] segment_files: original segment list files to be filtered
        :param Union[list, Path] filter_list: list used for filtering or a path to a text file containing the entries of
        that list one per line
        :param bool invert_match: black list (if False) or white list (if True) usage
        """
        assert isinstance(filter_list, tk.Path) or isinstance(filter_list, list)
        self.segment_files = segment_files
        self.filter_list = filter_list
        self.invert_match = invert_match

        num_segment_lists = len(self.segment_files)
        self.out_single_segment_files = dict(
            (i, self.output_path("segments.%d" % i))
            for i in range(1, num_segment_lists + 1)
        )
        self.out_segment_path = MultiOutputPath(
            self, "segments.$(TASK)", self.out_single_segment_files
        )

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        if isinstance(self.filter_list, tk.Path):
            filter_list = [
                line.rstrip() for line in open(self.filter_list.get_path(), "r")
            ]
        elif isinstance(self.filter_list, list):
            filter_list = self.filter_list
        else:
            assert False

        for idx, segment_file in self.segment_files.items():
            segment_list = [
                line.rstrip() for line in open(segment_file.get_path(), "r")
            ]
            non_empty = False
            with open(
                self.out_single_segment_files[idx].get_path(), "wt"
            ) as segment_file_filtered:
                for segment in segment_list:
                    if (self.invert_match and segment in filter_list) or (
                        not self.invert_match and segment not in filter_list
                    ):
                        segment_file_filtered.write(segment + "\n")
                        non_empty = True
            if not non_empty:
                logging.warning(
                    "Segment file empty after filtering: {}".format(
                        self.out_single_segment_files[idx].get_path()
                    )
                )


class FilterSegmentsByAlignmentConfidenceJob(Job):
    def __init__(
        self, alignment_logs, percentile, crp=None, plot=True, absolute_threshold=None
    ):
        """
        :param dict[int,Path] alignment_logs: alignment_job.out_log_file; task_id -> log_file
        :param float percentile: percent of alignment segments to keep. should be in (0,100]. for :func:`np.percentile`
        :param float absolute_threshold: alignments with score above this number are discarded
        :param Optional[rasr.crp.CommonRasrParameters] crp: used to set the number of output segments.
            if none, number of alignment log files is used instead.
        :param bool plot: plot the distribution of alignment scores
        """
        self.alignment_logs = alignment_logs  # alignment_job.log_file
        self.percentile = percentile
        self.absolute_threshold = absolute_threshold
        self.num_segments = len(alignment_logs) if crp is None else crp.concurrent
        self.plot = plot

        self.out_single_segment_files = dict(
            (i, self.output_path("segments.%d" % i))
            for i in range(1, self.num_segments + 1)
        )
        self.out_segment_path = MultiOutputPath(
            self, "segments.$(TASK)", self.out_single_segment_files
        )
        self.out_single_file = self.output_path("filtered.segments")
        if plot:
            self.out_plot_avg = self.output_path("score.png")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        segment_dict = {}
        for task_id, log_file in self.alignment_logs.items():
            logging.info("Reading: {}".format(log_file))
            file_path = tk.uncached_path(log_file)
            document = ET.parse(uopen(file_path))
            _seg_list = document.findall(".//segment")
            for seg in _seg_list:
                avg = seg.find(".//score/avg")
                segment_dict[seg.attrib["full-name"]] = float(avg.text)
            del document

        logging.info("Scores has {} entries.".format(len(segment_dict)))
        score_np = np.asarray(list(segment_dict.values()))
        logging.info(
            "Max {}; Min {}; Median {}".format(
                score_np.max(), score_np.min(), np.median(score_np)
            )
        )
        avg_score_threshold = np.percentile(score_np, self.percentile)
        if np.isnan(avg_score_threshold):
            avg_score_threshold = np.inf
        logging.info(
            "Avg Threshold is {} with percentile {}".format(
                avg_score_threshold, self.percentile
            )
        )
        if self.absolute_threshold is not None:
            avg_score_threshold = min(avg_score_threshold, self.absolute_threshold)
        logging.info("Threshold is {}".format(avg_score_threshold))

        if self.plot:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plot_percentile = np.percentile(score_np, 90)  # there can be huge outliers
            np.clip(score_np, 0, 200, out=score_np)
            plt.hist(score_np, bins=100, range=(0, 200))
            plt.xlabel("Average Maximum-Likelihood Score")
            plt.ylabel("Number of Segments")
            plt.title("Histogram of Alignment Scores")
            plt.savefig(fname=self.out_plot_avg.get_path())

        # Only keep segments that are below the threshold
        filtered_segments = [
            seg for seg, avg in segment_dict.items() if avg <= avg_score_threshold
        ]
        logging.info("Have {} entries after filtering.".format(len(filtered_segments)))

        for idx, segments in enumerate(chunks(filtered_segments, self.num_segments)):
            with open(
                self.out_single_segment_files[idx + 1].get_path(), "wt"
            ) as segment_file:
                for segment in segments:
                    segment_file.write(segment + "\n")

        with open(self.out_single_file.get_path(), "wt") as segment_file:
            for segment in filtered_segments:
                segment_file.write(segment + "\n")


class FilterCorpusBySegmentsJob(Job):
    def __init__(
        self, bliss_corpus, segment_file, compressed=False, invert_match=False
    ):
        """
        :param Path bliss_corpus:
        :param list[Path]|Path segment_file: a single segment file or a list of segment files
        :param bool compressed:
        :param bool invert_match:
        """
        self.bliss_corpus = bliss_corpus
        self.segment_file_list = (
            [segment_file] if isinstance(segment_file, tk.Path) else segment_file
        )
        self.invert_match = invert_match

        self.out_corpus = self.output_path("corpus.xml" + (".gz" if compressed else ""))

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):

        segments = []
        for seg in self.segment_file_list:
            with open(tk.uncached_path(seg)) as f:
                lines = f.readlines()
                segments += [l.strip() for l in lines]

        logging.info(
            "There are #{} segments in the segment list.".format(len(segments))
        )
        segments = set(segments)
        c = corpus.Corpus()
        c.load(tk.uncached_path(self.bliss_corpus))
        for rec in c.all_recordings():
            if self.invert_match:
                rec.segments = [
                    x
                    for x in rec.segments
                    if x.fullname() not in segments and x.name not in segments
                ]
            else:
                rec.segments = [
                    x
                    for x in rec.segments
                    if x.fullname() in segments or x.name in segments
                ]

        c.dump(tk.uncached_path(self.out_corpus))


class FilterCorpusRemoveUnknownWordSegmentsJob(Job):
    def __init__(self, bliss_corpus, bliss_lexicon, case_sensitive=False):
        """
        :param Path bliss_corpus:
        :param Path bliss_lexicon:
        :param bool case_sensitive:
        """
        self.corpus = bliss_corpus
        self.lexicon = bliss_lexicon
        self.case_sensitive = case_sensitive

        self.out_corpus = self.output_path("corpus.xml.gz", cached=True)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        def maybe_to_lower(s):
            return s if self.case_sensitive else s.lower()

        lex_path = tk.uncached_path(self.lexicon)
        open_func = gzip.open if lex_path.endswith(".gz") else open
        with open_func(lex_path, "rt") as f:
            lex_root = ET.parse(f)
        vocabulary = set(
            [
                maybe_to_lower(o.text.strip() if o.text else "")
                for o in lex_root.findall(".//orth")
            ]
        )

        c = corpus.Corpus()
        c.load(tk.uncached_path(self.corpus))

        def not_only_unknowns(corpus, recording, segment):
            """
            :param Corpus corpus:
            :param Recording recording:
            :param Segment segment:
            :return: whether the orth of segment contains at least one known word
            :rtype: bool
            """
            orth = segment.orth
            if not orth:
                return True
            words = [maybe_to_lower(o) for o in orth.strip().split(" ")]
            return not all(w not in vocabulary for w in words)

        c.filter_segments(not_only_unknowns)
        c.dump(self.out_corpus.get_path())


class FilterCorpusBySegmentDurationJob(Job):
    def __init__(self, bliss_corpus, min_duration=0.1, max_duration=120.0):
        """
        :param Path bliss_corpus: path of the corpus file
        :param float min_duration: minimum duration for a segment to keep (in seconds)
        :param float max_duration: maximum duration for a segment to keep (in seconds)
        """
        self.bliss_corpus = bliss_corpus
        self.min_duration = min_duration
        self.max_duration = max_duration

        self.out_corpus = self.output_path("corpus.xml.gz", cached=True)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        inf = float("inf")

        def good_duration(corpus, recording, segment):
            l = segment.end - segment.start
            if l == inf:
                return True
            else:
                return l >= self.min_duration and l <= self.max_duration

        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())
        c.filter_segments(good_duration)
        c.dump(self.out_corpus.get_path())
