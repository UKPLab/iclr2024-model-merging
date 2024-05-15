__all__ = ["StoreAllophonesJob", "DumpStateTyingJob"]

import shutil

from sisyphus import *

Path = setup_path(__package__)

import i6_core.rasr as rasr
import i6_core.util as util


class StoreAllophonesJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        num_single_state_monophones=1,
        extra_config=None,
        extra_post_config=None,
    ):
        self.set_vis_name("Store Allophones")

        self.config, self.post_config = StoreAllophonesJob.create_config(
            crp, extra_config, extra_post_config
        )
        self.exe = self.select_exe(crp.allophone_tool_exe, "allophone-tool")
        self.num_single_state_monophones = (
            num_single_state_monophones  # usually only silence and noise
        )

        self.out_log_file = self.log_file_output_path("store-allophones", crp, False)
        self.out_allophone_file = self.output_path("allophones")
        self.out_num_allophones = self.output_var("num_allophones")
        self.out_num_monophones = self.output_var("num_monophones")
        self.out_num_monophone_states = self.output_var("num_monophone_states")

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", mini_task=True)

    def create_files(self):
        self.write_config(self.config, self.post_config, "store-allophones.config")
        self.write_run_script(self.exe, "store-allophones.config")

    def run(self):
        self.run_script(1, self.out_log_file)
        shutil.move("allophones", self.out_allophone_file.get_path())

        with open(self.out_allophone_file.get_path(), "rt") as f:
            allophones = f.readlines()[1:]

        self.out_num_allophones.set(len(allophones))

        num_monophones = len(set(a.split("{")[0] for a in allophones))
        self.out_num_monophones.set(num_monophones)

        self.config._update(
            self.post_config
        )  # make it easier to access states-per-phone
        states_per_phone = (
            self.config.allophone_tool.acoustic_model.hmm.states_per_phone
        )
        num_monophone_states = (
            self.num_single_state_monophones
            + (num_monophones - self.num_single_state_monophones) * states_per_phone
        )
        self.out_num_monophone_states.set(num_monophone_states)

    def cleanup_before_run(self, cmd, retry, *args):
        util.backup_if_exists("store-allophones.log")

    @classmethod
    def create_config(cls, crp, extra_config, extra_post_config, **kwargs):
        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "acoustic_model": "allophone-tool.acoustic-model",
                "lexicon": "allophone-tool.lexicon",
            },
            parallelize=False,
        )

        config.allophone_tool.acoustic_model.allophones.store_to_file = "allophones"

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        return super().hash({"config": config, "exe": kwargs["crp"].allophone_tool_exe})


class DumpStateTyingJob(rasr.RasrCommand, Job):
    def __init__(self, crp, extra_config=None, extra_post_config=None):
        self.set_vis_name("Dump state-tying")

        self.config, self.post_config = DumpStateTyingJob.create_config(
            crp, extra_config, extra_post_config
        )
        self.exe = self.select_exe(crp.allophone_tool_exe, "allophone-tool")

        self.out_log_file = self.log_file_output_path("dump-state-tying", crp, False)
        self.out_state_tying = self.output_path("state-tying")

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", mini_task=True)

    def create_files(self):
        self.write_config(self.config, self.post_config, "dump-state-tying.config")
        self.write_run_script(self.exe, "dump-state-tying.config")

    def run(self):
        self.run_script(1, self.out_log_file)
        shutil.move("state-tying", self.out_state_tying.get_path())

    def cleanup_before_run(self, cmd, retry, *args):
        util.backup_if_exists("dump-state-tying.log")

    @classmethod
    def create_config(cls, crp, extra_config, extra_post_config, **kwargs):
        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "acoustic_model": "allophone-tool.acoustic-model",
                "lexicon": "allophone-tool.lexicon",
            },
            parallelize=False,
        )

        config.allophone_tool.dump_state_tying.channel = "state-tying-channel"
        config.allophone_tool.channels.state_tying_channel.append = False
        config.allophone_tool.channels.state_tying_channel.compressed = False
        config.allophone_tool.channels.state_tying_channel.file = "state-tying"
        config.allophone_tool.channels.state_tying_channel.unbuffered = False

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        return super().hash({"config": config, "exe": kwargs["crp"].allophone_tool_exe})
