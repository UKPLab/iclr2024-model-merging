__all__ = [
    "DownloadLibriSpeechCorpusJob",
    "DownloadLibriSpeechMetadataJob",
    "LibriSpeechCreateBlissCorpusJob",
]

import os
import shutil
import subprocess

from sisyphus import *

from i6_core.lib import corpus
from i6_core.util import uopen


class DownloadLibriSpeechCorpusJob(Job):
    """
    Download a part of the LibriSpeech corpus from
    https://www.openslr.org/resources/12
    and checks for file integrity via md5sum

    (see also: https://www.openslr.org/12/)

    To get the corpus metadata, use
    DownloadLibriSpeechMetadataJob

    self.out_corpus_folder links to the root of the speaker_id/chapter/*
    folder structure
    """

    def __init__(self, corpus_key):
        """
        :param str corpus_key: corpus identifier, e.g. "train-clean-100"
        """
        self.corpus_key = corpus_key

        assert corpus_key in [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]

        self.out_corpus_folder = self.output_path("%s" % self.corpus_key)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        subprocess.check_call(
            ["wget", "https://www.openslr.org/resources/12/md5sum.txt"]
        )
        subprocess.check_call(
            ["wget", "https://www.openslr.org/resources/12/%s.tar.gz" % self.corpus_key]
        )

        with open("md5sum.txt", "rt") as md5_in, open(
            "md5sum-%s.txt" % self.corpus_key, "wt"
        ) as md5_out:
            for line in md5_in:
                split = line.strip().split(" ")
                if split[-1].split(".")[0] == self.corpus_key:
                    md5_out.write(line)
                    break

        subprocess.check_call(
            ["md5sum", "--status", "-c", "md5sum-%s.txt" % self.corpus_key]
        )
        subprocess.check_call(
            [
                "tar",
                "-xf",
                "%s.tar.gz" % self.corpus_key,
                "-C",
                ".",
            ]
        )
        self._move_files()
        os.unlink("%s.tar.gz" % self.corpus_key)
        shutil.rmtree("LibriSpeech")

    def _move_files(self):
        shutil.move(
            "LibriSpeech/%s" % self.corpus_key, self.out_corpus_folder.get_path()
        )


class DownloadLibriSpeechMetadataJob(DownloadLibriSpeechCorpusJob):
    """
    Downloads the metadata file and checks for md5sum integrity

    Defines outputs for "SPEAKERS.TXT, CHAPTERS.TXT and BOOKS.TXT"
    """

    # noinspection PyMissingConstructor
    def __init__(self):
        self.corpus_key = "raw-metadata"

        self._out_archive_folder = self.output_path("LibriSpeech")

        self.out_speakers = self.output_path("SPEAKERS.TXT")
        self.out_chapters = self.output_path("CHAPTERS.TXT")
        self.out_books = self.output_path("BOOKS.TXT")

    def _move_files(self):
        shutil.move("LibriSpeech/SPEAKERS.TXT", self.out_speakers.get_path())
        shutil.move("LibriSpeech/CHAPTERS.TXT", self.out_chapters.get_path())
        shutil.move("LibriSpeech/BOOKS.TXT", self.out_books.get_path())


class LibriSpeechCreateBlissCorpusJob(Job):
    """
    Creates a Bliss corpus from a LibriSpeech corpus folder using the speaker information in addition

    Outputs a single bliss .xml.gz file
    """

    def __init__(self, corpus_folder, speaker_metadata):
        """
        :param Path corpus_folder: Path to a LibriSpeech corpus folder
        :param Path speaker_metadata: Path to SPEAKER.TXT file from the MetdataJob (out_speakers)
        """
        self.corpus_folder = corpus_folder
        self.speaker_metadata = speaker_metadata

        self.out_corpus = self.output_path("corpus.xml.gz")

        self._speakers = {}  # dict(key: id, value: [sex, subset, min, name]
        self._transcripts = []  # [dict(name, chapter, segment, orth, path)]

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        self._get_speakers()
        self._get_transcripts()

        c = corpus.Corpus()
        c.name = os.path.basename(self.corpus_folder.get_path())

        used_speaker_ids = set()  # store which speakers are used

        for transcript in self._transcripts:
            name = "{0}-{1}-{2:04d}".format(
                transcript["speaker_id"], transcript["chapter"], transcript["segment"]
            )
            recording = corpus.Recording()
            recording.name = name
            recording.speaker_name = transcript["speaker_id"]
            recording.audio = "{}/{}.flac".format(transcript["path"], name)

            used_speaker_ids.add(transcript["speaker_id"])

            segment = corpus.Segment()
            segment.name = name
            segment.start = 0
            segment.end = float("inf")
            segment.orth = transcript["orth"].strip()

            recording.segments.append(segment)
            c.recordings.append(recording)

        for speaker_id, speaker_info in sorted(self._speakers.items()):
            if speaker_id not in used_speaker_ids:
                continue
            speaker = corpus.Speaker()
            speaker.name = speaker_id
            speaker.attribs["gender"] = "male" if speaker_info[0] == "M" else "female"
            c.add_speaker(speaker)

        c.dump(self.out_corpus.get_path())

    def _get_speakers(self):
        """
        Extract the speakers from the SPEAKERS.TXT file
        """
        with uopen(self.speaker_metadata, "r") as speakersfile:
            for line in speakersfile:
                if line[0] == ";":
                    continue
                procline = list(map(str.strip, line.split("|")))
                self._speakers[int(procline[0])] = [
                    procline[1],
                    procline[2],
                    float(procline[3]),
                    procline[4],
                ]

    def _get_transcripts(self):
        """
        Traverse the folder structure and search for the *.trans.txt files and read the content
        """
        for dirpath, dirs, files in sorted(
            os.walk(self.corpus_folder.get_path(), followlinks=True)
        ):
            for file in files:
                if not file.endswith(".trans.txt"):
                    continue
                with uopen(os.path.join(dirpath, file), "r") as transcription:
                    for line in transcription:
                        line_t = list(map(str.strip, line.split(" ", 1)))
                        orth = line_t[1]
                        procline = line_t[0].split("-")
                        transcript = {
                            "speaker_id": int(procline[0]),
                            "chapter": int(procline[1]),
                            "segment": int(procline[2]),
                            "orth": orth,
                            "path": dirpath,
                        }
                        self._transcripts.append(transcript)
