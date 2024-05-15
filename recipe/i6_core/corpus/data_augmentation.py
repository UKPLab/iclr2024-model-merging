__all__ = ["SelfNoiseCorpusJob", "ChangeCorpusSpeedJob"]

import glob
import logging
import os
import random
import shutil

from sisyphus import *

from i6_core.lib import corpus

Path = setup_path(__package__)


class SelfNoiseCorpusJob(Job):
    """
    Add noise to each recording in the corpus.
    The noise consists of audio data from other recordings in the corpus and is reduced by the given SNR.
    Only supports .wav files

    WARNING: This Job uses /dev/shm for performance reasons, please be cautious
    """

    def __init__(self, bliss_corpus, snr, corpus_name, n_noise_tracks=1, seed=0):
        """

        :param Path bliss_corpus: Bliss corpus with wav files
        :param float snr: signal to noise ratio in db, positive values only
        :param str corpus_name: name of the new corpus
        :param int n_noise_tracks: number of random (parallel) utterances to add
        :param int seed: seed for random utterance selection
        """
        self.bliss_corpus = bliss_corpus
        self.snr = snr
        self.corpus_name = corpus_name
        self.n_noise_tracks = n_noise_tracks
        self.seed = seed

        assert (
            isinstance(self.n_noise_tracks, int) and self.n_noise_tracks >= 1
        ), "number of noise tracks must be a positive integer"
        assert self.snr >= 0, "please provide a positive SNR"

        self.out_audio_folder = self.output_path("audio/", directory=True)
        self.out_corpus = self.output_path("noised.xml.gz")
        self.out_segment_file = self.output_path("noised.segments")

        self.rqmt = {"time": 12, "cpu": 2}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        id = os.path.basename(self.job_id())
        if not os.path.isdir(f"/dev/shm/{id}"):
            os.mkdir(f"/dev/shm/{id}")
        c = corpus.Corpus()
        nc = corpus.Corpus()
        segment_file_names = []

        c.load(tk.uncached_path(self.bliss_corpus))
        nc.name = self.corpus_name
        nc.speakers = c.speakers
        nc.default_speaker = c.default_speaker
        nc.speaker_name = c.speaker_name

        logging.info("Random seed used: {}".format(self.seed))
        rng = random.Random(self.seed)

        # store index of last segment
        for r in c.recordings:
            max_seg_end = 0
            for s in r.segments:
                if s.end > max_seg_end:
                    max_seg_end = s.end
            r.max_seg_end = max_seg_end

        # select noise files for each recording
        for i, r in enumerate(c.recordings):
            audio_name = r.audio
            target_length = r.max_seg_end
            reverbed_audio_name = "noised_" + audio_name.split("/")[-1]

            # remove any possibly existing temporary recordings (otherwise ffmpeg will ask for override)
            for p in glob.iglob(f"/dev/shm/{id}/tmp_concat_*.wav"):
                os.unlink(p)

            for n in range(self.n_noise_tracks):
                noise_length = 0
                noise_audios = []

                while noise_length < target_length:
                    random_index = rng.randint(0, len(c.recordings) - 1)
                    while random_index == i:
                        random_index = random.randint(0, len(c.recordings) - 1)
                    noise_audios.append(c.recordings[random_index])
                    noise_length += c.recordings[random_index].max_seg_end

                # create temp noise file
                temp_noise_track_file = "/dev/shm/{id}/tmp_concat_%i.wav" % n

                self.sh(
                    "ffmpeg -hide_banner -loglevel panic -f concat -safe 0 -i <(%s) '%s'"
                    % (
                        " ".join(['echo "file %s";' % f.audio for f in noise_audios]),
                        temp_noise_track_file,
                    ),
                    except_return_codes=(1,),
                )

            if self.n_noise_tracks == 1:
                self.sh(
                    "ffmpeg -hide_banner  -i '%s' -i '/dev/shm/{id}/tmp_concat_0.wav' "
                    "-filter_complex '[1]volume=-{snr}dB[a];[0][a]amix=duration=first[out]' "
                    "-map '[out]' '{audio_out}/%s'" % (audio_name, reverbed_audio_name)
                )
            else:
                ffmpeg_head = "ffmpeg -hide_banner  -i '%s' " % audio_name
                noise_inputs = " ".join(
                    [
                        "-i '/dev/shm/%s/tmp_concat_%i.wav'" % (id, i)
                        for i in range(self.n_noise_tracks)
                    ]
                )
                filter_head = ' -filter_complex "'
                volume_reduction = (
                    ";".join(
                        [
                            "[%i]volume=-%idB[a%i]" % (i + 1, self.snr, i + 1)
                            for i in range(self.n_noise_tracks)
                        ]
                    )
                    + ";"
                )
                mixer = (
                    "[0]"
                    + "".join(["[a%i]" % i for i in range(1, self.n_noise_tracks + 1)])
                    + "amix=duration=first:inputs=%i[out]" % (self.n_noise_tracks + 1)
                )
                filter_tail = '" -map "[out]" "{audio_out}/%s"' % reverbed_audio_name
                command = (
                    ffmpeg_head
                    + noise_inputs
                    + filter_head
                    + volume_reduction
                    + mixer
                    + filter_tail
                )
                self.sh(command)

            nr = corpus.Recording()
            nr.name = r.name
            nr.segments = r.segments
            nr.speaker_name = r.speaker_name
            nr.default_speaker = r.default_speaker
            nr.speakers = r.speakers
            nr.audio = str(self.out_audio_folder) + "/" + reverbed_audio_name
            nc.add_recording(nr)
            for s in nr.segments:
                segment_file_names.append(nc.name + "/" + nr.name + "/" + s.name + "\n")

        nc.dump(self.out_corpus.get_path())

        with open(tk.uncached_path(self.out_segment_file), "w") as segments_outfile:
            segments_outfile.writelines(segment_file_names)

        shutil.rmtree(f"/dev/shm/{id}")


class ChangeCorpusSpeedJob(Job):
    """
    Changes the speed of all audio files in the corpus (shifting time AND frequency)
    """

    def __init__(self, bliss_corpus, corpus_name, speed_factor, base_frequency):
        """

        :param Path bliss_corpus: Bliss corpus
        :param str corpus_name: name of the new corpus
        :param float speed_factor: relative speed factor
        :param int base_frequency: sampling rate of the audio files
        """
        self.bliss_corpus = bliss_corpus
        self.speed_factor = speed_factor
        self.corpus_name = corpus_name
        self.base_frequency = base_frequency

        assert self.speed_factor > 0, "speed factor needs to be greater than zero"

        self.out_corpus = self.output_path("speed_perturbed.xml.gz")
        self.out_audio_folder = self.output_path("audio/")
        self.out_segment_file = self.output_path("speed_perturbed.segments")

        self.rqmt = {"time": 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        if not os.path.isdir(str(self.out_audio_folder)):
            self.sh("mkdir '{audio_out}'")
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
            perturbed_audio_name = "perturbed_" + r.audio.split("/")[-1]

            self.sh(
                "ffmpeg -hide_banner -i '%s' -filter:a \"asetrate={base_frequency}*{speed_factor}\" "
                "-ar {base_frequency} '{audio_out}/%s'"
                % (r.audio, perturbed_audio_name)
            )

            pr = corpus.Recording()
            pr.name = r.name
            pr.segments = r.segments
            pr.speaker_name = r.speaker_name
            pr.speakers = r.speakers
            pr.default_speaker = r.default_speaker
            pr.audio = str(self.out_audio_folder) + "/" + perturbed_audio_name
            nc.add_recording(pr)
            for s in pr.segments:
                segment_file_names.append(nc.name + "/" + pr.name + "/" + s.name)
                s.start /= self.speed_factor
                s.end /= self.speed_factor

        nc.dump(str(self.out_corpus))

        with open(str(self.out_segment_file), "w") as segments_outfile:
            segments_outfile.writelines(segment_file_names)
