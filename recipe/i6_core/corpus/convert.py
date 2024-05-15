__all__ = [
    "CorpusReplaceOrthFromReferenceCorpus",
    "CorpusReplaceOrthFromTxtJob",
    "CorpusToStmJob",
    "CorpusToTextDictJob",
    "CorpusToTxtJob",
]

import gzip
import itertools
import pprint
import re

from typing import Dict, List, Optional, Tuple, Union

from sisyphus import *

from i6_core.lib import corpus
from i6_core.util import uopen

Path = setup_path(__package__)


class CorpusReplaceOrthFromReferenceCorpus(Job):
    """
    Copies the orth tag from one corpus to another through matching segment names.
    """

    def __init__(self, bliss_corpus: Path, reference_bliss_corpus: Path):
        """
        :param bliss_corpus: Corpus in which the orth tag is to be replaced
        :param reference_bliss_corpus: Corpus from which the orth tag replacement is taken
        """
        self.bliss_corpus = bliss_corpus
        self.reference_bliss_corpus = reference_bliss_corpus

        self.out_corpus = self.output_path("corpus.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        orth_c = corpus.Corpus()
        orth_c.load(self.reference_bliss_corpus.get_path())

        orths = {}
        for s in orth_c.segments():
            orth = s.orth
            tag = s.fullname()
            orths[tag] = orth

        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        for s in c.segments():
            tag = s.fullname()
            assert tag in orths.keys(), "Segment %s not found in reference corpus" % tag
            s.orth = orths[tag]

        c.dump(self.out_corpus.get_path())


class CorpusReplaceOrthFromTxtJob(Job):
    """
    Merge raw text back into a bliss corpus
    """

    def __init__(self, bliss_corpus, text_file, segment_file=None):
        """
        :param Path bliss_corpus: Bliss corpus
        :param Path text_file: a raw or gzipped text file
        :param Path|None segment_file: only replace the segments as specified in the segment file
        """
        self.bliss_corpus = bliss_corpus
        self.text_file = text_file
        self.segment_file = segment_file

        self.out_corpus = self.output_path("corpus.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        if self.segment_file:
            with uopen(self.segment_file.get_path(), "rt") as f:
                segments_whitelist = set(
                    l.strip() for l in f.readlines() if len(l.strip()) > 0
                )
            segment_iterator = filter(
                lambda s: s.fullname() in segments_whitelist, c.segments()
            )
        else:
            segment_iterator = c.segments()

        with uopen(self.text_file, "rt") as f:
            for segment, line in itertools.zip_longest(segment_iterator, f):
                assert (
                    segment is not None
                ), "there were more text file lines than segments"
                assert line is not None, "there were less text file lines than segments"
                assert len(line) > 0
                segment.orth = line.strip()

        c.dump(self.out_corpus.get_path())


class CorpusToStmJob(Job):
    """
    Convert a Bliss corpus into a .stm file
    """

    __sis_hash_exclude__ = {"non_speech_tokens": None, "punctuation_tokens": None}

    def __init__(
        self,
        bliss_corpus: Path,
        *,
        exclude_non_speech: bool = True,
        non_speech_tokens: Optional[List[str]] = None,
        remove_punctuation: bool = True,
        punctuation_tokens: Optional[Union[str, List[str]]] = None,
        fix_whitespace: bool = True,
        name: str = "",
        tag_mapping: List[Tuple[Tuple[str, str, str], Dict[int, tk.Path]]] = (),
    ):
        """

        :param bliss_corpus: Path to Bliss corpus
        :param exclude_non_speech: non speech tokens should be removed
        :param non_speech_tokens: defines the list of non speech tokens
        :param remove_punctuation: should punctuation be removed
        :param punctuation_tokens: defines list/string of punctuation tokens
        :param fix_whitespace: should white space be fixed. !!!be aware that the corpus loading already fixes white space!!!
        :param name: new corpus name
        :param tag_mapping: 3-string tuple contains ("short name", "long name", "description") of each tag.
            and the Dict[int, tk.Path] is e.g. the out_single_segment_files of a FilterSegments*Jobs
        """
        self.set_vis_name("Extract STM from Corpus")

        self.bliss_corpus = bliss_corpus
        self.exclude_non_speech = exclude_non_speech
        self.non_speech_tokens = (
            non_speech_tokens if non_speech_tokens is not None else []
        )
        self.remove_punctuation = remove_punctuation
        self.punctuation_tokens = (
            punctuation_tokens if punctuation_tokens is not None else []
        )
        self.fix_whitespace = fix_whitespace
        self.tag_mapping = tag_mapping
        self.name = name

        self.out_stm_path = self.output_path("%scorpus.stm" % name)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        tag_map = {}

        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        all_tags = [
            ("d%d" % i, "default%d" % i, "all other segments of category %d" % i)
            for i in range(len(self.tag_mapping) + 1)
        ]

        for segment in c.segments():
            tag_map[segment.fullname()] = [
                "d%d" % i for i in range(len(self.tag_mapping) + 1)
            ]

        for i, (tag, segments) in enumerate(self.tag_mapping):
            all_tags.append(tag)
            for file in segments.values():
                for segment in uopen(file):
                    if segment.rstrip() in tag_map:
                        tag_map[segment.rstrip()][i] = tag[0]

        with uopen(self.out_stm_path, "wt") as out:
            for segment in c.segments():
                speaker_name = (
                    segment.speaker().name
                    if segment.speaker() is not None
                    else segment.recording.name
                )
                segment_track = segment.track + 1 if segment.track else 1

                orth = f" {segment.orth.strip()} "

                if self.exclude_non_speech:
                    for nst in self.non_speech_tokens:
                        orth = self.replace_recursive(orth, nst)

                if self.remove_punctuation:
                    for pt in self.punctuation_tokens:
                        orth = orth.replace(pt, "")

                if self.fix_whitespace:
                    orth = re.sub(" +", " ", orth)

                orth = orth.strip()

                out.write(
                    "%s %d %s %5.2f %5.2f <%s> %s\n"
                    % (
                        segment.recording.name,
                        segment_track,
                        "_".join(speaker_name.split()),
                        segment.start,
                        segment.end,
                        ",".join(tag_map[segment.fullname()]),
                        orth,
                    )
                )
            for tag in all_tags:
                out.write(';; LABEL "%s" "%s" "%s"\n' % tag)

    @classmethod
    def replace_recursive(cls, orthography, token):
        """
        recursion is required to find repeated tokens
        string.replace is not sufficient
        some other solution might also work
        """
        pos = orthography.find(f" {token} ")
        if pos == -1:
            return orthography
        else:
            orthography = orthography.replace(f" {token} ", " ")
            return cls.replace_recursive(orthography, token)


class CorpusToTextDictJob(Job):
    """
    Extract the Text from a Bliss corpus to fit a "{key: text}" structure (e.g. for RETURNN)
    """

    def __init__(self, bliss_corpus, segment_file=None, invert_match=False):
        """
        :param Path bliss_corpus: bliss corpus file
        :param Path|None segment_file: a segment file as optional whitelist
        :param bool invert_match: use segment file as blacklist (needs to contain full segment names then)
        """
        self.bliss_corpus = bliss_corpus
        self.segment_file = segment_file
        self.invert_match = invert_match

        self.out_dictionary = self.output_path("text_dictionary.py")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        dictionary = {}

        segments = None
        if self.segment_file:
            with uopen(self.segment_file) as f:
                segments = set(line.decode().strip() for line in f)

        for segment in c.segments():
            orth = segment.orth.strip()
            key = segment.fullname()
            if segments:
                if (
                    not self.invert_match
                    and key not in segments
                    and segment.name not in segments
                ):
                    continue
                if self.invert_match and key in segments:
                    continue
            dictionary[key] = orth

        dictionary_string = pprint.pformat(dictionary, width=1000)
        with uopen(self.out_dictionary, "wt") as f:
            f.write(dictionary_string)


class CorpusToTxtJob(Job):
    """
    Extract orth from a Bliss corpus and store as raw txt or gzipped txt
    """

    def __init__(self, bliss_corpus, segment_file=None, gzip=False):
        """

        :param Path bliss_corpus: Bliss corpus
        :param Path segment_file: segment file
        :param bool gzip: gzip the output text file
        """
        self.set_vis_name("Extract TXT from Corpus")

        self.bliss_corpus = bliss_corpus
        self.gzip = gzip
        self.segment_file = segment_file

        self.out_txt = self.output_path("corpus.txt" + (".gz" if gzip else ""))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        if self.segment_file:
            with uopen(self.segment_file, "rt") as f:
                segments_whitelist = set(
                    l.strip() for l in f.readlines() if len(l.strip()) > 0
                )
        else:
            segments_whitelist = None

        with uopen(self.out_txt.get_path(), "wt") as f:
            for segment in c.segments():
                if (not segments_whitelist) or (
                    segment.fullname() in segments_whitelist
                ):
                    f.write(segment.orth + "\n")
