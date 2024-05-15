__all__ = ["ViterbiTdpTuningJob", "TdpFromAlignmentJob"]

import numpy as np
import json
import os
import subprocess

from sisyphus import *

Path = setup_path(__package__)

from .alignment import AlignmentJob
import i6_core.util as util


class ViterbiTdpTuningJob(Job):
    def __init__(
        self, crp, feature_flow, feature_scorer, allophone_files, am_args, max_iter=5
    ):
        self.flow = feature_flow
        self.scorer = feature_scorer
        self.allophone = allophone_files
        self.max_iter = max_iter
        self.cur_iter = 0
        self.am_args = am_args
        self.crp = crp
        self.last_tdp = None

        self.am_args_opt = self.output_var("am_args_opt")
        self.transition_prob = self.output_path("trans.prob")
        self.log = self.output_path("prob.log")

        self.tdp_list = []

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        self.am_args_opt.set(self.last_tdp.am_args)

        from shutil import copyfile

        copyfile(
            self.last_tdp.transition_prob.get_path(), self.transition_prob.get_path()
        )

        with open(self.log.get_path(), "w") as of:
            json.dump(self.tdp_list, of)

    def update(self):
        if self.cur_iter < self.max_iter:

            if self.last_tdp and self.last_tdp._sis_finished():
                self.am_args = util.get_val(self.last_tdp.am_args)
                self.tdp_list.append(self.am_args)

            self.crp.acoustic_model_config.tdp["*"].loop = self.am_args[
                "tdp_transition"
            ][0]
            self.crp.acoustic_model_config.tdp["*"].forward = self.am_args[
                "tdp_transition"
            ][1]
            self.crp.acoustic_model_config.tdp["*"].skip = self.am_args[
                "tdp_transition"
            ][2]

            self.crp.acoustic_model_config.tdp.silence.loop = self.am_args[
                "tdp_silence"
            ][0]
            self.crp.acoustic_model_config.tdp.silence.forward = self.am_args[
                "tdp_silence"
            ][1]

            self.tune_tdp()

    def tune_tdp(self):
        align = AlignmentJob(
            crp=self.crp, feature_flow=self.flow, feature_scorer=self.scorer
        )
        self.add_input(align.out_alignment_bundle)
        tdp = TdpFromAlignmentJob(
            crp=self.crp, alignment=align, allophones=self.allophone
        )
        self.add_input(tdp.transition_prob)
        self.last_tdp = tdp
        self.cur_iter += 1


class TdpFromAlignmentJob(Job):
    """
    Alignments look like when views in archiver:
    time=  7  emission=  17420  allophone=  w{#+iy}@i  index=  17420  state=  0
    time=  8  emission=  17420  allophone=  w{#+iy}@i  index=  17420  state=  0
    time=  9  emission=  67126284  allophone=  w{#+iy}@i  index=  17420  state=  1
    time=  10  emission=  134235148  allophone=  w{#+iy}@i  index=  17420  state=  2
    time=  11  emission=  134235148  allophone=  w{#+iy}@i  index=  17420  state=  2
    state are extracted and loops, forward and skip transitions are counted
    """

    def __init__(self, crp, alignment, allophones):
        self.crp = crp
        self.alignment = alignment
        self.allophones = allophones
        self.concurrent = crp.concurrent
        self.rqmt = {"time": 1, "cpu": 1, "gpu": 0, "mem": 1}

        self.transition_count = self.output_path("trans.count")
        self.transition_prob = self.output_path("trans.prob")

        self.am_args = self.output_var("am_args")

    def tasks(self):
        yield Task(
            "run", resume="run", rqmt=self.rqmt, args=range(1, self.concurrent + 1)
        )
        yield Task("prob", resume="prob", mini_task=True)

    def run(self, task_id):
        alignment_path = self.alignment.out_single_alignment_caches[task_id].get_path()
        segment_path = self.crp.segment_path.hidden_paths[task_id].get_path()
        exe = os.path.join(
            gs.RASR_ROOT,
            "arch",
            gs.RASR_ARCH,
            "%s.%s" % ("archiver", gs.RASR_ARCH),
        )

        trans_dict = {"phon": {0: 0, 1: 0, 2: 0}, "si": {0: 0, 1: 0, 2: 0}}

        with open(segment_path, "r") as segment_file:
            for seg in segment_file:
                seg = seg.strip("\n")
                args = [
                    exe,
                    "--allophone-file",
                    self.allophones.get_path(),
                    "--mode",
                    "show",
                    "--type",
                    "align",
                    alignment_path,
                    seg,
                ]
                res = subprocess.run(args, stdout=subprocess.PIPE)
                lines = res.stdout.decode("utf-8").split("\n")

                transistions = {"phon": [], "si": []}

                for l in lines:
                    if "<" in l:
                        continue
                    l.replace("\n", "")
                    parts = l.split("\t")
                    if len(parts) < 10:
                        continue
                    state = int(parts[-1])
                    allophone = parts[5]
                    if "#" in allophone:
                        transistions["si"].append(state)
                    else:
                        transistions["phon"].append(state)

                trans_mapping = np.asarray([[0, 1, 2], [2, 0, 1], [1, 2, 0]])

                for phon, states in transistions.items():
                    prev = 0
                    lsf = []
                    if len(states) > 0:
                        for t in states:
                            n = trans_mapping[prev, t]
                            lsf.append(n)
                            prev = t

                    unique, counts = np.unique(lsf, return_counts=True)
                    r = dict(zip(unique.astype(np.float), counts.astype(np.float)))
                    for trans, count in r.items():
                        trans_dict[phon][trans] += count

        with open(
            self.transition_count.get_path() + ".{}".format(task_id), "w"
        ) as out_file:
            out_file.write(json.dumps(trans_dict))

    def prob(self):
        trans_dict = {"phon": {"0": 0, "1": 0, "2": 0}, "si": {"0": 0, "1": 0, "2": 0}}

        for i in range(1, self.concurrent + 1):
            with open(
                self.transition_count.get_path() + ".{}".format(i), "rb"
            ) as in_file:
                d = json.load(in_file)
                for phon, sub_dict in d.items():
                    for trans, count in sub_dict.items():
                        trans_dict[phon][trans] += count

        for phon, sub_dict in trans_dict.items():
            N = 0
            for trans, count in sub_dict.items():
                N += count

            for trans in sub_dict:
                sub_dict[trans] /= N

        am_args = {
            "tdp_transition": (
                -np.log(trans_dict["phon"]["0"]),
                -np.log(trans_dict["phon"]["1"]),
                -np.log(trans_dict["phon"]["2"]),
                20.0,
            ),
            "tdp_silence": (
                -np.log(trans_dict["si"]["0"]),
                -np.log(trans_dict["si"]["1"]),
                "infinity",
                20.0,
            ),
        }
        self.am_args.set(am_args)

        with open(self.transition_prob.get_path(), "w") as out_file:
            out_file.write(json.dumps(trans_dict))
