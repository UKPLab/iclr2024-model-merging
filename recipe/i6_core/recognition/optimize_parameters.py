__all__ = ["OptimizeAMandLMScaleJob"]

import collections
import os
import subprocess as sp

from sisyphus import *

Path = setup_path(__package__)

import i6_core.rasr as rasr
import i6_core.util as util


class OptimizeAMandLMScaleJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        lattice_cache,
        initial_am_scale,
        initial_lm_scale,
        scorer_cls,
        scorer_kwargs,
        scorer_hyp_param_name="hyp",
        maxiter=100,
        precision=2,
        opt_only_lm_scale=False,
        extra_config=None,
        extra_post_config=None,
    ):
        self.set_vis_name("Optimize AM+LM score")

        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = OptimizeAMandLMScaleJob.create_config(**kwargs)
        self.exe = self.select_exe(crp.flf_tool_exe, "flf-tool")
        self.initial_am_scale = initial_am_scale
        self.initial_lm_scale = initial_lm_scale
        self.lattice_cache = lattice_cache
        self.maxiter = maxiter
        self.precision = precision
        self.opt_only_lm_scale = opt_only_lm_scale
        self.scorer_cls = scorer_cls
        self.scorer_kwargs = scorer_kwargs
        self.scorer_hyp_param_name = scorer_hyp_param_name

        self.out_log_file = self.output_path("optimization.log")
        self.out_best_am_score = self.output_var("bast_am_score")  # contains typo
        self.out_best_lm_score = self.output_var("bast_lm_score")  # contains typo

        self.rqmt = {"time": 6, "cpu": 1, "mem": 1}

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def create_files(self):
        self.write_config(self.config, self.post_config, "rescore.config")
        self.write_run_script(self.exe, "rescore.config")

    def run(self):
        am_scale = self.initial_am_scale
        lm_scale = self.initial_lm_scale

        result_cache = collections.OrderedDict()

        def calc_wer(params):
            def clip_float(f):
                return "%.*f" % (self.precision, f)

            params = list(map(lambda x: round(x, self.precision), params))
            if self.opt_only_lm_scale:
                am_scale = self.initial_am_scale
                lm_scale = params[0]
            else:
                am_scale, lm_scale = params
            if am_scale < 0.0 or lm_scale < 0.0:
                return 100.0
            am_str, lm_str = clip_float(am_scale), clip_float(lm_scale)
            if (am_str, lm_str) in result_cache:
                return result_cache[(am_str, lm_str)]

            ctm_file = "result.am-%s.lm-%s.ctm" % (am_str, lm_str)
            log_file = "log.am-%s.lm-%s.log" % (am_str, lm_str)
            if os.path.exists(ctm_file + ".gz"):
                sp.check_call(["gunzip", ctm_file + ".gz"])
            else:
                self.run_script(
                    1,
                    log_file,
                    args=[
                        "--am-scale=%s" % am_str,
                        "--lm-scale=%s" % lm_str,
                        "--ctm-file=%s" % ctm_file,
                    ],
                )

            scorer_kwargs = dict(**self.scorer_kwargs)
            scorer_kwargs[self.scorer_hyp_param_name] = tk.Path(
                os.path.abspath(ctm_file)
            )
            scorer = self.scorer_cls(**scorer_kwargs)
            wer = scorer.calc_wer()

            sp.check_call(["gzip", ctm_file])

            result_cache[(am_str, lm_str)] = wer
            print("AM: %s LM: %s WER: %f" % (am_str, lm_str, wer))

            return wer

        if self.opt_only_lm_scale:
            x0 = [lm_scale]
        else:
            x0 = [am_scale, lm_scale]

        import scipy.optimize

        xopt, fopt, direc, iter, funccalls, warnflag = scipy.optimize.fmin_powell(
            func=calc_wer,
            x0=x0,
            maxiter=self.maxiter,
            xtol=10**-self.precision,
            ftol=10**-self.precision,
            full_output=True,
        )

        if self.opt_only_lm_scale:
            lm_scale = xopt
        else:
            am_scale = xopt[0]
            lm_scale = xopt[1]
        self.out_best_am_score.set(float(am_scale))
        self.out_best_lm_score.set(float(lm_scale))
        with open(self.out_log_file.get_path(), "wt") as f:
            f.write(
                "Found optimum at am-scale = %f lm-scale = %f with WER %f\n"
                % (am_scale, lm_scale, fopt)
            )
            f.write("%d iterations\n" % iter)
            f.write("%d funccalls\n" % funccalls)
            for (am_scale, lm_scale), wer in result_cache.items():
                f.write("%s %s %f\n" % (am_scale, lm_scale, wer))

    def cleanup_before_run(self, cmd, retry, *args):
        util.backup_if_exists("lm_and_state_tree.log")

    @classmethod
    def create_config(
        cls, crp, lattice_cache, extra_config, extra_post_config, **kwargs
    ):
        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "corpus": "flf-lattice-tool.corpus",
                "lexicon": "flf-lattice-tool.lexicon",
            },
        )

        config.flf_lattice_tool.network.initial_nodes = "speech-segment"

        config.flf_lattice_tool.network.speech_segment.type = "speech-segment"
        config.flf_lattice_tool.network.speech_segment.links = (
            "0->archive-reader:1 0->dump-ctm:1"
        )

        config.flf_lattice_tool.network.archive_reader.type = "archive-reader"
        config.flf_lattice_tool.network.archive_reader.links = "scale-lm"
        config.flf_lattice_tool.network.archive_reader.format = "flf"
        config.flf_lattice_tool.network.archive_reader.path = lattice_cache
        config.flf_lattice_tool.network.archive_reader.flf.partial.keys = "am lm"

        config.flf_lattice_tool.network.scale_lm.type = "rescale"
        config.flf_lattice_tool.network.scale_lm.links = "scale-pronunciation"
        config.flf_lattice_tool.network.scale_lm.lm.scale = "$(lm-scale)"
        config.flf_lattice_tool.network.scale_lm.am.scale = 1.0

        config.flf_lattice_tool.network.scale_pronunciation.type = (
            "extend-by-pronunciation-score"
        )
        config.flf_lattice_tool.network.scale_pronunciation.links = "to-lemma"
        config.flf_lattice_tool.network.scale_pronunciation.key = "am"
        config.flf_lattice_tool.network.scale_pronunciation.scale = "$(am-scale)"
        config.flf_lattice_tool.network.scale_pronunciation.rescore_mode = (
            "in-place-cached"
        )

        config.flf_lattice_tool.network.to_lemma.type = "map-alphabet"
        config.flf_lattice_tool.network.to_lemma.links = "best"
        config.flf_lattice_tool.network.to_lemma.map_input = "to-lemma"
        config.flf_lattice_tool.network.to_lemma.project_input = True

        config.flf_lattice_tool.network.best.type = "best"
        config.flf_lattice_tool.network.best.links = "dump-ctm"
        config.flf_lattice_tool.network.best.algorithm = "bellman-ford"

        config.flf_lattice_tool.network.dump_ctm.type = "dump-traceback"
        config.flf_lattice_tool.network.dump_ctm.links = "sink"
        config.flf_lattice_tool.network.dump_ctm.format = "ctm"
        config.flf_lattice_tool.network.dump_ctm.dump.channel = "$(ctm-file)"
        post_config.flf_lattice_tool.network.dump_ctm.ctm.fill_empty_segments = True

        config.flf_lattice_tool.network.sink.type = "sink"
        config.flf_lattice_tool.network.sink.warn_on_empty_lattice = True
        config.flf_lattice_tool.network.sink.error_on_empty_lattice = False

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        d = {
            "config": config,
            "exe": kwargs["crp"].flf_tool_exe,
            "initial_am_scale": kwargs["initial_am_scale"],
            "initial_lm_scale": kwargs["initial_lm_scale"],
            "lattice_cache": kwargs["lattice_cache"],
            "scorer_cls": kwargs["scorer_cls"],
            "scorer_kwargs": kwargs["scorer_kwargs"],
        }
        if kwargs["opt_only_lm_scale"]:
            d["opt_only_lm_scale"] = kwargs["opt_only_lm_scale"]

        return super().hash(d)
