__all__ = ["BlissFfmpegJob"]

import copy
import logging
import os
import subprocess

from sisyphus import *

from i6_core.lib import corpus


class BlissFfmpegJob(Job):
    """
    Applies an FFMPEG audio filter to all recordings of a bliss corpus.
    This Job is extremely generic, as any valid audio option/filter string will work.
    Please consider using more specific jobs that use this Job as super class,
    see e.g. BlissChangeEncodingJob

    WARNING:
        - This job assumes that file names of individual recordings are unique across the whole corpus.
        - Do not change the duration of the audio files when you have multiple segments per audio,
          as the segment information will be incorrect afterwards.

    Typical applications:

    **Changing Audio Format/Encoding**

        - specify in `output_format` what container you want to use. If
          the filter string is empty (""), ffmepg will automatically use a default encoding option

        - specify specific encoding with `-c:a <codec>`. For a list of available codecs
          and their options see https://ffmpeg.org/ffmpeg-codecs.html#Audio-Encoders

        - specify a fixed bitrate with `-b:a <bit_rate>`, e.g. `64k`. Variable bitrate options depend on the
          used encoder, refer to the online documentation in this case

        - specify a sample rate with `-ar <sample_rate>`. FFMPEG will do proper resampling,
          so the speed of the audio is NOT changed.


    **Changing Channel Layout**

        - for detailed informations see https://trac.ffmpeg.org/wiki/AudioChannelManipulation

        - convert to mono `-ac 1`

        - selecting a specific audio channel:
          `-filter_complex [0:a]channelsplit=channel_layout=stereo:channels=FR[right] -map [right]`
          For a list of channels/layouts use `ffmpeg -layouts`


    **Simple Filter Syntax**

    For a list of available filters see: https://ffmpeg.org/ffmpeg-filters.html

    `-af <filter_name>=<first_param>=<first_param_value>:<second_param>=<second_param_value>`


    **Complex Filter Syntax**

    `-filter_complex [<input>]<simple_syntax>[<output>];[<input>]<simple_syntax>[<output>];...`

    Inputs and outputs can be namend arbitrarily, but the default stream 0 audio can be accessed with [0:a]

    The output stream that should be written into the audio is defined with `-map [<output_stream>]`

    IMPORTANT! Do not forget to add and escape additional quotation marks correctly
    for parameters to`-af` or `-filter_complex`

    """

    def __init__(
        self,
        corpus_file,
        ffmpeg_options=None,
        recover_duration=True,
        output_format=None,
        ffmpeg_binary=None,
        hash_binary=False,
    ):
        """

        :param Path corpus_file: bliss corpus
        :param list(str)|None ffmpeg_options: list of additional ffmpeg parameters
        :param bool recover_duration: if the filter changes the duration of the audio, set to True
        :param str output_format: output file ending to determine container format (without dot)
        :param Path|str|None ffmpeg_binary: path to a ffmpeg binary, uses system "ffmpeg" if None
        :param bool hash_binary: In some cases it might be required to work with a specific ffmpeg version,
                                 in which case the binary needs to be hashed

        """
        self.corpus_file = corpus_file
        self.ffmpeg_options = ffmpeg_options
        self.recover_duration = recover_duration
        self.output_format = output_format
        self.ffmpeg_binary = ffmpeg_binary if ffmpeg_binary else "ffmpeg"
        self.hash_binary = hash_binary

        self.out_audio_folder = self.output_path("audio/", directory=True)
        self.out_corpus = self.output_path("corpus.xml.gz")

        self.rqmt = {"time": 4, "cpu": 4, "mem": 8}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)
        if self.recover_duration:
            # recovering is not multi-threaded, so force cpu=1
            recover_rqmt = copy.copy(self.rqmt)
            recover_rqmt["cpu"] = 1
            yield Task("run_recover_duration", rqmt=recover_rqmt)

    def run(self):
        c = corpus.Corpus()
        c.load(tk.uncached_path(self.corpus_file))

        from multiprocessing import pool

        p = pool.Pool(self.rqmt["cpu"])
        p.map(self._perform_ffmpeg, c.recordings)

        for r in c.recordings:
            audio_filename = self._get_output_filename(r)
            r.audio = os.path.join(self.out_audio_folder.get_path(), audio_filename)

        if self.recover_duration:
            c.dump("temp_corpus.xml.gz")
        else:
            c.dump(tk.uncached_path(self.out_corpus))

    def run_recover_duration(self):
        """
        Open all files with "soundfile" and extract the length information

        :return:
        """
        import soundfile

        c = corpus.Corpus()
        c.load("temp_corpus.xml.gz")

        for r in c.all_recordings():
            assert len(r.segments) == 1, "needs to be a single segment recording"
            old_duration = r.segments[0].end
            data, sample_rate = soundfile.read(open(r.audio, "rb"))
            new_duration = len(data) / sample_rate
            logging.info(
                "%s: adjusted from %f to %f seconds"
                % (r.segments[0].name, old_duration, new_duration)
            )
            r.segments[0].end = new_duration

        c.dump(self.out_corpus.get_path())

    def _get_output_filename(self, recording):
        """
        returns a new audio filename with a potentially
        changed file ending based on "output_format"

        :param recording:
        :return:
        :rtype str
        """
        audio_filename = os.path.basename(recording.audio)
        if self.output_format is not None:
            name, ext = os.path.splitext(audio_filename)
            audio_filename = name + "." + self.output_format
        return audio_filename

    def _perform_ffmpeg(self, recording):
        """
        Build and call an FFMPEG command to apply on a recording

        :param corpus.Recording recording:
        :return:
        """
        audio_filename = self._get_output_filename(recording)

        target = os.path.join(self.out_audio_folder.get_path(), audio_filename)
        if not os.path.exists(target):
            logging.info("try converting %s" % target)
            command_head = [
                self.ffmpeg_binary,
                "-hide_banner",
                "-y",
                "-i",
                recording.audio,
            ]
            command_tail = [
                os.path.join(self.out_audio_folder.get_path(), audio_filename)
            ]
            if self.ffmpeg_options is None or len(self.ffmpeg_options) == 0:
                command = command_head + command_tail
            else:
                command = command_head + self.ffmpeg_options + command_tail
            subprocess.check_call(command)
        else:
            logging.info("skipped existing %s" % target)

    @classmethod
    def hash(cls, kwargs):
        d = copy.copy(kwargs)
        if not kwargs["hash_binary"]:
            d.pop("ffmpeg_binary")
        return super().hash(d)
