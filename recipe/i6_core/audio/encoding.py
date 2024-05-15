__all__ = ["BlissChangeEncodingJob"]

from sisyphus import Path

from i6_core.audio.ffmpeg import BlissFfmpegJob


class BlissChangeEncodingJob(BlissFfmpegJob):
    """
    Uses ffmpeg to convert all audio files of a bliss corpus (file format, encoding, channel layout)
    """

    __sis_hash_exclude__ = {"recover_duration": None}

    def __init__(
        self,
        corpus_file,
        output_format,
        sample_rate=None,
        codec=None,
        codec_options=None,
        fixed_bitrate=None,
        force_num_channels=None,
        select_channels=None,
        ffmpeg_binary=None,
        hash_binary=False,
        recover_duration=None,
    ):
        """
        For all parameter holds that "None" means to use the ffmpeg defaults, which depend on the input file
        and the output format specified.

        :param Path corpus_file: bliss corpus
        :param str|None output_format: output file ending to determine container format (without dot)
        :param int|None sample_rate: target sample rate of the audio
        :param str|None codec: specify the codec, codecs are listed with `ffmpeg -codecs`
        :param list(str)|None codec_options: specify additional codec specific options
            (be aware of potential conflicts with "fixed bitrate" and "sample_rate")
        :param int|str|None: fixed_bitrate: a target bitrate (be aware that not all codecs support all bitrates)
        :param int|None force_num_channels: specify the channel number, exceeding channels will be merged
        :param tuple(str)|None select_channels: tuple of (channel_layout, channel_name), see `ffmpeg -layouts`
            this is useful if the new encoding might have an effect on the duration, or if no duration was specified
            in the source corpus
        :param Path|str|None ffmpeg_binary: path to a ffmpeg binary, uses system "ffmpeg" if None
        :param bool hash_binary: In some cases it might be required to work with a specific ffmpeg version,
                                 in which case the binary needs to be hashed
        :param bool|None recover_duration: This will open all files with "soundfile" and extract the length information.
            There might be minimal differences when converting the encoding, so only set this to `False` if you're
            willing to accept this risk. `None` (default) means that the duration is recovered if either `output_format`
            or `codec` is specified because this might possibly lead to duration mismatches.
        """
        ffmpeg_options = []

        if select_channels:
            assert isinstance(select_channels, tuple) and len(select_channels) == 2
            ffmpeg_options += [
                "-filter_complex",
                "[0:a]channelsplit=channel_layout=%s:channels=%s[out]"
                % select_channels,
                "-map",
                "[out]",
            ]
        if codec:
            ffmpeg_options += ["-c:a", codec]

        if codec_options:
            ffmpeg_options += codec_options

        if fixed_bitrate:
            ffmpeg_options += ["-b:a", str(fixed_bitrate)]

        if sample_rate:
            ffmpeg_options += ["-ar", str(sample_rate)]

        if force_num_channels:
            ffmpeg_options += ["-ac", str(force_num_channels)]

        if recover_duration is None:
            if output_format is None and codec is None:
                recover_duration = False
            else:
                recover_duration = True

        super().__init__(
            corpus_file=corpus_file,
            ffmpeg_options=ffmpeg_options,
            recover_duration=recover_duration,
            output_format=output_format,
            ffmpeg_binary=ffmpeg_binary,
            hash_binary=hash_binary,
        )
