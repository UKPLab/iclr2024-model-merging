"""
Switchboard is conversational telephony speech with 8 Khz audio files. The training data consists of
300h hours.
Reference: https://catalog.ldc.upenn.edu/LDC97S62

number of recordings: 4876
number of segments: 249624
number of speakers: 2260
"""


from sisyphus import *

from collections import defaultdict
import glob
import subprocess
import shutil
import os
import re

from i6_core.lib import corpus
from i6_core.util import uopen
from i6_core.tools.download import DownloadJob


def _map_token(token):
    """
    This function applies some mapping rules for Switchboard transcription words
    Reference: https://github.com/espnet/espnet/blob/master/egs/swbd/asr1/local/swbd1_map_words.pl

    :param str token: string representing token, e.g word
    :rtype str
    """

    mapped_token = re.sub(
        "(|\-)^\[laughter-(.+)\](|\-)$", "\g<1>\g<2>\g<3>", token
    )  # e.g. [laughter-story] -> story;
    # 1 and 3 relate to preserving trailing "-"
    mapped_token = re.sub(
        "^\[(.+)/.+\](|\-)$", "\g<1>\g<2>", mapped_token
    )  # e.g. [it'n/isn't] -> it'n ... note
    # 1st part may include partial-word stuff, which we process further below,
    # e.g. [LEM[GUINI]-/LINGUINI]
    # the (|\_) at the end is to accept and preserve trailing -'s.
    mapped_token = re.sub(
        "^(|\-)\[[^][]+\](.+)$", "-\g<2>", mapped_token
    )  # e.g. -[an]y , note \047 is quote;
    # let the leading - be optional on input, as sometimes omitted.
    mapped_token = re.sub(
        "^(.+)\[[^][]+\](|\-)$", "\g<1>-", mapped_token
    )  # e.g. ab[solute]- -> ab-;
    # let the trailing - be optional on input, as sometimes omitted.
    mapped_token = re.sub(
        "([^][]+)\[.+\]$", "\g<1>", mapped_token
    )  # e.g. ex[specially]-/especially] -> ex-
    # which is a  mistake in the input.
    mapped_token = re.sub(
        "^\{(.+)\}$", "\g<1>", mapped_token
    )  # e.g. {yuppiedom} -> yuppiedom
    mapped_token = re.sub(
        "([a-z])\[([^][])+\]([a-z])", "\g<1>-\g<3>", mapped_token
    )  # e.g. ammu[n]it- -> ammu-it-
    mapped_token = re.sub("_\d$", "", mapped_token)  # e.g. them_1 -> them

    return mapped_token


class DownloadSwitchboardTranscriptionAndDictJob(Job):
    """
    Downloads switchboard training transcriptions and dictionary (or lexicon)
    """

    def __init__(self):
        self.out_raw_dict = self.output_path("swb_trans/sw-ms98-dict.text")
        self.out_trans_dir = self.output_path("swb_trans")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        zipped_filename = "switchboard_word_alignments.tar.gz"
        subprocess.check_call(
            ["wget", "http://www.openslr.org/resources/5/" + zipped_filename]
        )
        subprocess.check_call(
            [
                "tar",
                "-xf",
                zipped_filename,
                "-C",
                ".",
            ]
        )
        shutil.move("swb_ms98_transcriptions", self.out_trans_dir.get_path())
        os.remove(zipped_filename)


class DownloadSwitchboardSpeakersStatsJob(DownloadJob):
    """
    Note that this does not contain the speaker info for all recordings. We assume later that each
    recording has a unique speaker and a unique id is used for those recordings with unknown speakers info
    """

    def __init__(self):
        super(DownloadSwitchboardSpeakersStatsJob, self).__init__(
            url="http://www.isip.piconepress.com/projects/switchboard/doc/statistics/ws97_speaker_stats.text",
            checksum="64f538839073dbbdf46027fff40cec57a11c5de1eed4e8b22b50ed86038d9e90",
        )

    @classmethod
    def hash(cls, parsed_args):
        return Job.hash(parsed_args)


class CreateSwitchboardSpeakersListJob(Job):
    """
    Given some speakers statistics info, this job creates a text file having on each line:
        speaker_id gender recording
    """

    def __init__(self, speakers_stats_file):
        """
        :param tk.Path speakers_stats_file: speakers stats text file
        """
        self.speakers_stats_file = speakers_stats_file
        self.out_speakers_list = self.output_path("speakers_list.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        speaker_id, gender, rec_name = None, None, None
        with uopen(self.speakers_stats_file) as read_f, uopen(
            self.out_speakers_list, "w"
        ) as out_f:
            for line in read_f:
                l = line.strip().split()
                if len(l) < 2:
                    continue
                if l[1] == "F" or l[1] == "M":  # start new speaker
                    speaker_id = l[0]
                    gender = l[1]
                    rec_name = l[2]
                elif l[0].endswith("A") or l[0].endswith("B"):  # recording name
                    rec_name = l[0]
                else:
                    continue

                if speaker_id:
                    out_f.write(
                        speaker_id + " " + gender + " " + rec_name + "\n"
                    )  # speaker_id gender recording


class CreateSwitchboardBlissCorpusJob(Job):
    """
    Creates Switchboard bliss corpus xml

    segment name format: sw2001B-ms98-a-<folder-name>
    """

    def __init__(self, audio_dir, trans_dir, speakers_list_file):
        """
        :param tk.Path audio_dir: path for audio data
        :param tk.Path trans_dir: path for transcription data. see `DownloadSwitchboardTranscriptionAndDictJob`
        :param tk.Path speakers_list_file: path to a speakers list text file with format:
                speaker_id gender recording
            on each line. see `CreateSwitchboardSpeakersListJob` job
        """
        self.audio_dir = audio_dir
        self.trans_dir = trans_dir
        self.speakers_list_file = speakers_list_file
        self.out_corpus = self.output_path("swb.corpus.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        c = corpus.Corpus()
        c.name = "switchboard-1"

        rec_to_segs = self._get_rec_to_segs_map()

        rec_to_speaker = {}
        with uopen(self.speakers_list_file) as f:
            for line in f:
                l = line.strip().split()
                assert len(l) == 3
                assert (
                    l[2] not in rec_to_speaker
                ), "duplicate recording name: {}?".format(l[2])
                assert l[1] in ["F", "M"]

                # "sw0" prefix is added to match recording names
                rec_to_speaker["sw0" + l[2]] = {
                    "speaker_id": l[0],
                    "gender": {"M": "male", "F": "female"}.get(l[1]),
                }

        # assume unique speaker for each recording with no speaker info
        unk_spk_id = 1
        for rec in sorted(rec_to_segs.keys()):
            if rec not in rec_to_speaker:
                rec_to_speaker[rec] = {"speaker_id": "speaker#" + str(unk_spk_id)}
                unk_spk_id += 1

        for rec_name, segs in sorted(rec_to_segs.items()):
            recording = corpus.Recording()
            recording.name = rec_name
            recording.audio = os.path.join(self.audio_dir.get_path(), rec_name + ".wav")

            assert os.path.exists(
                recording.audio
            ), "recording {} does not exist?".format(recording.audio)

            assert (
                rec_name in rec_to_speaker
            ), "recording {} does not have speaker id?".format(rec_name)
            rec_speaker_id = rec_to_speaker[rec_name]["speaker_id"]

            for seg in segs:
                segment = corpus.Segment()
                segment.name = seg[0]
                segment.start = float(seg[1])
                segment.end = float(seg[2])
                segment.speaker_name = rec_speaker_id
                segment.orth = self._filter_orth(seg[3])
                if len(segment.orth) == 0:
                    continue

                recording.segments.append(segment)
            c.recordings.append(recording)

        # add speakers to corpus
        for speaker_info in rec_to_speaker.values():
            speaker = corpus.Speaker()
            speaker.name = speaker_info["speaker_id"]
            if speaker_info.get("gender", None):
                speaker.attribs["gender"] = speaker_info["gender"]
            c.add_speaker(speaker)

        c.dump(self.out_corpus.get_path())

    @staticmethod
    def _filter_orth(orth):
        """
        Filters orth by handling special cases such as silence tag removal, partial words, etc

        :param str orth: segment orth to be preprocessed
        """
        special_tokens = {
            "[vocalized-noise]",
            "[noise]",
            "[laughter]",
        }
        removed_tokens = {
            "[silence]",
            "<b_aside>",
            "<e_aside>",
        }  # unnecessary tags to be removed
        filtered_orth = []
        tokens = orth.strip().split()
        for token_ in tokens:
            token = token_.strip()
            if token in removed_tokens:
                continue
            elif token in special_tokens:
                filtered_orth.append(
                    token.upper()
                )  # make upper case for consistency with older setups
            else:
                filtered_orth.append(_map_token(token))

        # do not add empty transcription segments
        all_special = True
        for token in filtered_orth:
            if token.lower() not in special_tokens:
                all_special = False
                break
        if all_special:
            return ""

        out = " ".join(filtered_orth)

        # replace &
        out = out.replace("AT&T's", "AT and T")
        out = out.replace("&", " and ")

        return out

    def _get_rec_to_segs_map(self):
        """
        Returns recording to list of segments mapping
        """
        rec_to_segs = defaultdict(list)
        for trans_file in glob.glob(
            os.path.join(self.trans_dir.get_path(), "*/*/*-trans.text")
        ):
            with uopen(trans_file, "rt") as f:
                for line in f:
                    seg_info = line.strip().split(" ", 3)  # name start end orth
                    assert len(seg_info) == 4
                    rec_name = (
                        seg_info[0].split("-")[0].replace("sw", "sw0")
                    )  # e.g: sw2001A-ms98-a-0022 -> sw02001A
                    rec_to_segs[rec_name].append(seg_info)
        return rec_to_segs


class CreateSwitchboardLexiconTextFileJob(Job):
    """
    This job creates SWB preprocessed dictionary text file consistent with the training corpus given a raw dictionary
    text file downloaded within the transcription directory using `DownloadSwitchboardTranscriptionAndDictJob` Job.
    The resulted dictionary text file will be passed as argument to `LexiconFromTextFileJob` job in order to create
    bliss xml lexicon.
    """

    def __init__(self, raw_dict_file):
        """
        :param tk.Path raw_dict_file: path containing the raw dictionary text file
        """
        self.raw_dict_file = raw_dict_file
        self.out_dict = self.output_path("dict.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(self.raw_dict_file) as read_f, uopen(self.out_dict, "w") as out_f:
            for line in read_f.readlines()[1:]:
                if line.startswith("#"):  # skip comment
                    continue
                parts = line.strip().split(" ", 1)
                if len(parts) < 2:
                    continue
                token = parts[0].replace("&amp;", "&")  # e.g A&amp;E -> A&E
                mapped_token = _map_token(token)  # preprocessing as corpus
                out_f.write(mapped_token + " " + parts[1] + "\n")
