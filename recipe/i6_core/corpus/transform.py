__all__ = [
    "ReplaceTranscriptionFromCtmJob",
    "AddCacheToCorpusJob",
    "CompressCorpusJob",
    "MergeCorporaJob",
    "MergeStrategy",
    "MergeCorpusSegmentsAndAudioJob",
    "ShiftCorpusSegmentStartJob",
    "ApplyLexiconToCorpusJob",
]

import bisect
import collections
import enum
import logging
import math
import os
import wave
import xml.etree.cElementTree as ET

from i6_core.lib import corpus, lexicon
from i6_core.util import uopen

from sisyphus import *

Path = setup_path(__package__)


class ReplaceTranscriptionFromCtmJob(Job):
    def __init__(self, bliss_corpus, ctm_path, remove_empty_segments=True):
        self.set_vis_name("Replace Transcription from CTM file")

        self.bliss_corpus = bliss_corpus
        self.ctm_path = ctm_path
        self.remove_empty_segments = remove_empty_segments

        gzip_output = tk.uncached_path(bliss_corpus).endswith(".gz")
        self.output_corpus_path = self.output_path(
            "corpus.xml" + (".gz" if gzip_output else ""), cached=True
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        transcriptions = collections.defaultdict(list)
        with open(tk.uncached_path(self.ctm_path), "rt") as f:
            for line in f:
                if line.startswith(";;"):
                    continue

                fields = line.split()
                if 5 <= len(fields) <= 6:
                    recording = fields[0]
                    start = float(fields[2])
                    word = fields[4]
                    transcriptions[recording].append((start, word))

        for recording, times_and_words in transcriptions.items():
            times_and_words.sort()

        corpus_path = tk.uncached_path(self.bliss_corpus)
        c = corpus.Corpus()
        c.load(corpus_path)

        recordings_to_delete = []

        for recording in c.all_recordings():
            times = [s[0] for s in transcriptions[recording.name]]
            words = [s[1] for s in transcriptions[recording.name]]

            if len(words) == 0 and self.remove_empty_segments:
                recordings_to_delete = recording
                continue

            segments_to_delete = []
            for idx, segment in enumerate(recording.segments):
                left_idx = bisect.bisect_left(times, segment.start)
                right_idx = bisect.bisect_left(times, segment.end)

                if left_idx == right_idx and self.remove_empty_segments:
                    segments_to_delete.append(idx)
                    continue

                segment.orth = " ".join(words[left_idx:right_idx]).replace("&", "&amp;")

            for sidx in reversed(segments_to_delete):
                del recording.segments[sidx]

        c.dump(self.output_corpus_path.get_path())


class AddCacheToCorpusJob(Job):
    """
    Adds cache manager call to all audio paths in a corpus file
    :param Path bliss_corpus: bliss corpora file path
    """

    def __init__(self, bliss_corpus):
        self.bliss_corpus = bliss_corpus
        self.cached_corpus = self.output_path("corpus.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        c = corpus.Corpus()
        c.load(tk.uncached_path(self.bliss_corpus))
        for recording in c.all_recordings():
            recording.audio = gs.file_caching(recording.audio)
        c.dump(tk.uncached_path(self.cached_corpus))


class CompressCorpusJob(Job):
    """
    Compresses a corpus by concatenating audio files and using a compression codec.
    Does currently not support corpora with subcorpora, files need to be .wav
    :param Path bliss_corpus: path to an xml corpus file with wave recordings
    :param str format: supported file formats, currently limited to mp3
    :param str bitrate: bitrate as string, e.g. '32k' or '192k', can also be an integer e.g. 192000
    :param int max_num_splits: maximum number of resulting audio files.
    """

    def __init__(self, bliss_corpus, format="mp3", bitrate="32k", max_num_splits=15):
        self.bliss_corpus = bliss_corpus
        self.num_splits = max_num_splits
        self.format = format
        self.bitrate = str(bitrate)

        assert bitrate[-1] == "k" or len(bitrate) >= 4, "your bitrate seems to low %s"
        assert format in ["mp3", "wav", "ogg"], "untested format %s" % format

        self.compressed_corpus = self.output_path("corpus.xml.gz")
        self.segment_map = self.output_path("segment_map.xml.gz")
        self.audio_folder = self.output_path("audio", directory=True)

        self.rqmt = {"time": 16, "mem": 8, "cpu": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        assert (
            len(c.subcorpora) == 0
        ), "CompressCorpus is not working for corpus files containing subcorpora"

        # for each recording, extract duration
        total_duration = self.add_duration_to_recordings(c)

        # print useful information
        logging.info(f"corpus name {c.name}")
        logging.info(f"number of recordings: {len(c.recordings)}")
        logging.info(f"total duration: {total_duration} sec")

        # determine split
        split_duration = total_duration / float(self.num_splits)
        logging.info(f"split duration: {split_duration} sec")

        # create new compressed corpus file
        cc = corpus.Corpus()
        cc.name = c.name
        cc.speaker_name = c.speaker_name
        cc.speakers = c.speakers
        cc.default_speaker = c.default_speaker

        sm = corpus.SegmentMap()

        # temporary store of recordings
        split_recordings = []
        current_duration = 0
        current_split_index = 0

        # segment count for verification
        segment_count = 0

        for i, recording in enumerate(c.recordings):
            # append recording and its duration to the l
            split_recordings.append(recording)
            current_duration += recording.duration

            # now we have all recordings in the duration for a single file or it is the last recording
            if current_duration > split_duration or i + 1 == len(c.recordings):
                new_recording_element = corpus.Recording()

                split_name = "split_%i" % current_split_index
                logging.info(
                    f"storing split {split_name} with duration {current_duration}"
                )

                new_recording_element.name = split_name
                output_path = os.path.join(
                    self.audio_folder.get_path(), f"{split_name}.{self.format}"
                )
                new_recording_element.audio = output_path
                current_timestamp = 0

                # store all audio paths that are to be concatenated for a split
                ffmpeg_inputs = []

                for split_recording in split_recordings:
                    recording_name = split_recording.name
                    for j, segment in enumerate(split_recording.segments):
                        # update the segment times based on the current time
                        segment.start = float(segment.start) + current_timestamp

                        # segment ends can be inf, use the duration of the recording in that case
                        if segment.end == "inf":
                            segment.end = split_recording.duration + current_timestamp
                        else:
                            segment.end = float(segment.end) + current_timestamp

                        # add segment keymap entry
                        sm_entry = corpus.SegmentMapItem()
                        # add original name to key
                        sm_entry.key = "/".join([c.name, recording_name, segment.name])

                        # if a segment has no name, use a 1-based index
                        # of the form corpus_name/split_i/original_recording_name#segment_j
                        # otherwise create entries in the form corpus_name/split_i/original_recording_name#segment_name
                        if segment.name is None:
                            segment.name = recording_name + "#" + str(j + 1)
                        else:
                            segment.name = recording_name + "#" + segment.name

                        # add new name as segment map value
                        sm_entry.value = "/".join([c.name, split_name, segment.name])
                        sm.map_entries.append(sm_entry)

                        new_recording_element.segments.append(segment)
                        segment_count += 1

                    # update the time stamp with the recording length and add to ffmpeg merge list
                    current_timestamp += split_recording.duration
                    ffmpeg_inputs.append(split_recording.audio)

                # run ffmpeg and add the new recording
                self.run_ffmpeg(ffmpeg_inputs, output_path)
                cc.add_recording(new_recording_element)

                # reset variables
                current_split_index += 1
                split_recordings = []
                current_duration = 0

        logging.info(f"segment count: {segment_count}")
        cc.dump(tk.uncached_path(self.compressed_corpus))

        sm.dump(tk.uncached_path(self.segment_map))

    def add_duration_to_recordings(self, c):
        """
        open each recording, extract the duration and add the duration to the recording object
        # TODO: this is a lengthy operation, but so far there was no alternative...
        :param corpus.Corpus c:
        :return:
        """
        total_duration = 0
        for recording in c.recordings:
            audio_path = recording.audio
            assert audio_path.endswith(
                ".wav"
            ), "compress corpus can only operate on .wav files"
            wave_header = wave.open(open(audio_path, "rb"))
            duration = float(wave_header.getnframes()) / float(
                wave_header.getframerate()
            )
            recording.duration = duration
            total_duration += duration

        return total_duration

    def run_ffmpeg(self, ffmpeg_inputs, output_path):
        # run ffmpeg
        with open("filelist.txt", "w") as command_file:
            command_file.write(" ".join(["file '%s'\n" % f for f in ffmpeg_inputs]))

        if self.format == "wav":
            self.sh("ffmpeg -f concat -safe 0 -i filelist.txt '%s'" % output_path)
        else:
            self.sh(
                "ffmpeg -f concat -safe 0 -i filelist.txt -b:a '%s' '%s'"
                % (self.bitrate, output_path)
            )

    def info(self):
        """
        read the log.run file to extract the current status of the compression job
        :return:
        """
        basepath = self._sis_path()
        if os.path.isfile(os.path.join(basepath, "/log.run.1")):
            info_str = self.sh(
                "cat "
                + basepath
                + '/log.run.1| grep -o "split_[0-9]*" | tail -n 1 | grep -o "[0-9]*" ',
                capture_output=True,
                sis_quiet=True,
                except_return_codes=(
                    141,
                    1,
                ),
            )
            if len(info_str) == 0:
                info_str = "0"
            return "split " + str(info_str).strip() + " / " + str(self.num_splits)
        return None


class MergeStrategy(enum.Enum):
    SUBCORPORA = 0
    FLAT = 1
    CONCATENATE = 2


class MergeCorporaJob(Job):
    """
    Merges Bliss Corpora files into a single file as subcorpora or flat
    """

    def __init__(self, bliss_corpora, name, merge_strategy=MergeStrategy.SUBCORPORA):
        """
        :param Iterable[Path] bliss_corpora: any iterable of bliss corpora file paths to merge
        :param str name: name of the new corpus (subcorpora will keep the original names)
        :param MergeStrategy merge_strategy: how the corpora should be merged, e.g. as subcorpora or flat
        """
        self.bliss_corpora = bliss_corpora
        self.name = name
        self.merge_strategy = merge_strategy

        self.out_merged_corpus = self.output_path("merged.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        merged_corpus = corpus.Corpus()
        merged_corpus.name = self.name
        for corpus_path in self.bliss_corpora:
            c = corpus.Corpus()
            c.load(tk.uncached_path(corpus_path))
            if self.merge_strategy == MergeStrategy.SUBCORPORA:
                merged_corpus.add_subcorpus(c)
            elif self.merge_strategy == MergeStrategy.FLAT:
                for rec in c.all_recordings():
                    merged_corpus.add_recording(rec)
                merged_corpus.speakers.update(c.speakers)
            elif self.merge_strategy == MergeStrategy.CONCATENATE:
                for subcorpus in c.top_level_subcorpora():
                    merged_corpus.add_subcorpus(subcorpus)
                for rec in c.top_level_recordings():
                    merged_corpus.add_recording(rec)
                for speaker in c.top_level_speakers():
                    merged_corpus.add_speaker(speaker)
            else:
                assert False, "invalid merge strategy"

        merged_corpus.dump(self.out_merged_corpus.get_path())


class MergeCorpusSegmentsAndAudioJob(Job):
    """
    This job merges segments and audio files based on a rasr cluster map and a list of cluster_names.
    The cluster map should map segments to something like cluster.XXX where XXX is a natural number (starting with 1).
    The lines in the cluster_names file will be used as names for the recordings in the new corpus.

    The job outputs a new corpus file + the corresponding audio files.
    """

    def __init__(self, bliss_corpus, cluster_map, cluster_names):
        self.corpus_file = bliss_corpus
        self.cluster_map = cluster_map
        self.cluster_names = cluster_names

        self.output_corpus = self.output_path("output.corpus.xml.gz", cached=True)
        self.audio_output = self.output_path("audio", True)

        self.rqmt = {"cpu": 1, "time": 1.0, "mem": 2.0}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        with uopen(self.cluster_names, "rt") as f:
            cluster_names = [l.strip() for l in f]

        clusters = collections.defaultdict(set)
        with uopen(self.cluster_map, "rt") as f:
            t = ET.parse(f)
            for mi in t.findall(".//map-item"):
                k = mi.attrib["key"]
                v = int(mi.attrib["value"].split(".")[-1]) - 1
                clusters[cluster_names[v]].add(k)

        c = corpus.Corpus()
        c.load(tk.uncached_path(self.corpus_file))

        original_segments = {}
        for s in c.segments():
            original_segments[s.fullname()] = s

        audio = {}
        transcriptions = {}
        for cluster_name in clusters:
            clusters[cluster_name] = list(sorted(clusters[cluster_name]))
            transcriptions[cluster_name] = " ".join(
                original_segments[s].orth for s in clusters[cluster_name]
            )
            audio[cluster_name] = [
                (r.audio, s.start, s.end)
                for n in clusters[cluster_name]
                for s, r in [(original_segments[n], original_segments[n].recording)]
            ]

        new_c = corpus.Corpus()
        new_c.name = c.name
        for cluster_name, audio_files in audio.items():
            out_path = os.path.join(self.audio_output.get_path(), cluster_name + ".wav")
            if os.path.exists(out_path):
                os.unlink(out_path)
            with open(f"{cluster_name}.txt", "wt") as f:
                for af in audio_files:
                    f.write(f"file {af[0]}\ninpoint {af[1]}\n")
                    if not math.isinf(af[2]):
                        f.write(f"outpoint {af[2]}\n")
            self.sh(
                f"ffmpeg -loglevel fatal -hide_banner -f concat -safe 0 -i '{cluster_name}.txt' '{out_path}'"
            )

            r = corpus.Recording()
            r.name = cluster_name
            r.audio = out_path
            s = corpus.Segment()
            s.name = "1"
            s.start = 0.0
            s.end = float("inf")
            s.orth = transcriptions[cluster_name]
            r.add_segment(s)

            new_c.add_recording(r)

        new_c.dump(self.output_corpus.get_path())


class ShiftCorpusSegmentStartJob(Job):
    """
    Shifts the start time of a corpus to change the fft window offset
    """

    def __init__(self, bliss_corpus, corpus_name, shift):
        """
        :param Path bliss_corpus: path to a bliss corpus file
        :param str corpus_name: name of the new corpus
        :param int shift: shift in seconds
        """
        self.bliss_corpus = bliss_corpus
        self.corpus_name = corpus_name
        self.shift = shift
        self.out_shifted_corpus = self.output_path("shifted.xml.gz")
        self.out_segments = self.output_path("shifted.segments")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        c = corpus.Corpus()
        nc = corpus.Corpus()
        segment_file_names = []

        c.load(tk.uncached_path(self.bliss_corpus))
        nc.name = self.corpus_name
        nc.speakers = c.speakers
        nc.default_speaker = c.default_speaker
        nc.speaker_name = c.speaker_name
        # store index of last segment
        for r in c.recordings:
            sr = corpus.Recording()
            sr.name = r.name
            sr.segments = r.segments
            sr.speaker_name = r.speaker_name
            sr.speakers = r.speakers
            sr.default_speaker = r.default_speaker
            sr.audio = r.audio
            nc.add_recording(sr)
            for s in sr.segments:
                segment_file_names.append(nc.name + "/" + sr.name + "/" + s.name)
                s.start += self.shift

        nc.dump(str(self.out_shifted_corpus))

        with open(str(self.out_segments), "w") as segments_outfile:
            segments_outfile.writelines(segment_file_names)


class LexiconStrategy(enum.Enum):
    PICK_FIRST = 0


class ApplyLexiconToCorpusJob(Job):
    """
    Use a bliss lexicon to convert all words in a bliss corpus into their phoneme representation.

    Currently only supports picking the first phoneme.
    """

    def __init__(
        self,
        bliss_corpus,
        bliss_lexicon,
        word_separation_orth=None,
        strategy=LexiconStrategy.PICK_FIRST,
    ):
        """
        :param Path bliss_corpus: path to a bliss corpus xml
        :param Path bliss_lexicon: path to a bliss lexicon file
        :param str|None word_separation_orth: a default word separation lemma orth. The corresponding phoneme
            (or phonemes in some special cases) are inserted between each word.
            Usually it makes sense to use something like "[SILENCE]" or "[space]" or so).
        :param LexiconStrategy strategy: strategy to determine which representation is selected
        """
        self.bliss_corpus = bliss_corpus
        self.bliss_lexicon = bliss_lexicon
        self.word_separation_orth = word_separation_orth
        self.strategy = strategy

        self.out_corpus = self.output_path("corpus.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lex = lexicon.Lexicon()
        lex.load(self.bliss_lexicon.get_path())

        # build lookup dict
        lookup_dict = {}
        for lemma in lex.lemmata:
            for orth in lemma.orth:
                if orth and self.strategy == LexiconStrategy.PICK_FIRST:
                    if len(lemma.phon) > 0:
                        lookup_dict[orth] = lemma.phon[0]

        if self.word_separation_orth is not None:
            word_separation_phon = lookup_dict[self.word_separation_orth]
            print("using word separation symbol: %s" % word_separation_phon)
            separator = " %s " % word_separation_phon
        else:
            separator = " "

        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        for segment in c.segments():
            try:
                words = [lookup_dict[w] for w in segment.orth.split(" ")]
                segment.orth = separator.join(words)
            except LookupError:
                raise LookupError(
                    "Out-of-vocabulary word detected, please make sure that there are no OOVs remaining by e.g. applying G2P"
                )

        c.dump(self.out_corpus.get_path())
