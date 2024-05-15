__all__ = ["DownloadLJSpeechCorpusJob", "LJSpeechCreateBlissCorpusJob"]

import os
import shutil
import subprocess
import wave

from sisyphus import *

from i6_core.lib import corpus
from i6_core.util import uopen, check_file_sha256_checksum


class DownloadLJSpeechCorpusJob(Job):
    """
    Downloads, checks and extracts the LJSpeech corpus.
    """

    def __init__(self):
        self.out_ljspeech_folder = self.output_path("LJSpeech-1.1")
        self.out_audio_folder = self.output_path("LJSpeech-1.1/wavs")
        self.out_metadata = self.output_path("LJSpeech-1.1/metadata.csv")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        subprocess.check_call(
            ["wget", "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"]
        )
        check_file_sha256_checksum(
            "LJSpeech-1.1.tar.bz2",
            "be1a30453f28eb8dd26af4101ae40cbf2c50413b1bb21936cbcdc6fae3de8aa5",
        )
        subprocess.check_call(["tar", "-xvf", "LJSpeech-1.1.tar.bz2"])
        shutil.move("LJSpeech-1.1", self.out_ljspeech_folder.get_path())
        os.unlink("LJSpeech-1.1.tar.bz2")


class LJSpeechCreateBlissCorpusJob(Job):
    """
    Generate a Bliss xml from the downloaded LJspeech dataset
    """

    def __init__(self, metadata, audio_folder, name="LJSpeech"):
        """

        :param Path metadata: path to metadata.csv
        :param Path audio_folder: path to the wavs folder
        :param name: overwrite default corpus name
        """

        self.metadata = metadata
        self.audio_folder = audio_folder
        self.name = name

        self.out_bliss_corpus = self.output_path("ljspeech.xml.gz")

    def tasks(self):
        yield Task("run", rqmt={"time": 8, "mem": 2})

    def run(self):
        c = corpus.Corpus()
        c.name = self.name

        with uopen(self.metadata, "rt") as metadata_file:
            for line in metadata_file:
                name, text, processed_text = line.split("|")
                audio_file_path = os.path.join(
                    self.audio_folder.get_path(), name + ".wav"
                )
                assert os.path.isfile(
                    audio_file_path
                ), "Audio file %s was not found in provided audio path %s" % (
                    audio_file_path,
                    self.audio_folder.get_path(),
                )

                recording = corpus.Recording()
                recording.name = name
                recording.audio = audio_file_path
                segment = corpus.Segment()
                segment.orth = processed_text.strip()
                segment.name = name

                wave_info = wave.open(audio_file_path)
                segment.start = 0
                segment.end = wave_info.getnframes() / wave_info.getframerate()
                wave_info.close()

                recording.add_segment(segment)
                c.add_recording(recording)

        c.dump(self.out_bliss_corpus.get_path())
