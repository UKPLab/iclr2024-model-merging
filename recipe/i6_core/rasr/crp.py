from sisyphus.http_server import object_to_html
from sisyphus import tk

from .command import RasrCommand
from .config import RasrConfig


class CommonRasrParameters:
    """
    This class holds often used parameters for Rasr.
    """

    def __init__(self, base=None):
        """
        :param CommonRasrParameters|None base:
        """
        self.base = base
        if base is None:
            self.acoustic_model_config = None
            self.acoustic_model_post_config = None
            self.corpus_config = None
            self.corpus_post_config = None
            self.lexicon_config = None
            self.lexicon_post_config = None
            self.language_model_config = None
            self.language_model_post_config = None
            self.recognizer_config = None
            self.recognizer_post_config = None
            self.log_config = None
            self.log_post_config = None
            self.compress_log_file = True
            self.default_log_channel = "stderr"

            self.audio_format = "wav"
            self.corpus_duration = 1.0
            self.concurrent = 1
            self.segment_path = None

            self.acoustic_model_trainer_exe = None
            self.allophone_tool_exe = None
            self.costa_exe = None
            self.feature_extraction_exe = None
            self.feature_statistics_exe = None
            self.flf_tool_exe = None
            self.kws_tool_exe = None
            self.lattice_processor_exe = None
            self.lm_util_exe = None
            self.nn_trainer_exe = None
            self.speech_recognizer_exe = None

            self.python_home = None
            self.python_program_name = None

    def __getattr__(self, name):
        if super().__getattribute__("base") is not None and hasattr(self.base, name):
            return getattr(self.base, name)
        raise AttributeError(name)

    def __repr__(self):
        return str(self.__dict__)

    def html(self):
        return object_to_html(self.__dict__)

    def set_executables(self, rasr_binary_path, rasr_arch="linux-x86_64-standard"):
        """
        Set all executables to a specific binary folder path

        :param tk.Path rasr_binary_path: path to the rasr binary folder
        :param str rasr_arch: RASR compile architecture suffix
        :return:
        """
        assert isinstance(rasr_binary_path, tk.Path)
        self.acoustic_model_trainer_exe = rasr_binary_path.join_right(
            f"acoustic-model-trainer.{rasr_arch}"
        )
        self.allophone_tool_exe = rasr_binary_path.join_right(
            f"allophone-tool.{rasr_arch}"
        )
        self.costa_exe = rasr_binary_path.join_right(f"costa.{rasr_arch}")
        self.feature_extraction_exe = rasr_binary_path.join_right(
            f"feature-extraction.{rasr_arch}"
        )
        self.feature_statistics_exe = rasr_binary_path.join_right(
            f"feature-statistics.{rasr_arch}"
        )
        self.flf_tool_exe = rasr_binary_path.join_right(f"flf-tool.{rasr_arch}")
        self.kws_tool_exe = None  # does not exist
        self.lattice_processor_exe = rasr_binary_path.join_right(
            f"lattice-processor.{rasr_arch}"
        )
        self.lm_util_exe = rasr_binary_path.join_right(f"lm-util.{rasr_arch}")
        self.nn_trainer_exe = rasr_binary_path.join_right(f"nn-trainer.{rasr_arch}")
        self.speech_recognizer_exe = rasr_binary_path.join_right(
            f"speech-recognizer.{rasr_arch}"
        )


def crp_add_default_output(
    crp, compress=False, append=False, unbuffered=False, compress_after_run=True
):
    """
    :param CommonRasrParameters crp:
    :param bool compress:
    :param bool append:
    :param bool unbuffered:
    :param bool compress_after_run:
    """
    if compress:
        compress_after_run = False

    config = RasrConfig()
    config["*"].configuration.channel = "output-channel"
    config["*"].real_time_factor.channel = "output-channel"
    config["*"].system_info.channel = "output-channel"
    config["*"].time.channel = "output-channel"
    config["*"].version.channel = "output-channel"

    config["*"].log.channel = "output-channel"
    config["*"].warning.channel = "output-channel, stderr"
    config["*"].error.channel = "output-channel, stderr"

    config["*"].statistics.channel = "output-channel"
    config["*"].progress.channel = "output-channel"
    config["*"].dot.channel = "nil"

    post_config = RasrConfig()
    post_config["*"].encoding = "UTF-8"
    post_config["*"].output_channel.file = "$(LOGFILE)" + (".gz" if compress else "")
    post_config["*"].output_channel.compressed = compress
    post_config["*"].output_channel.append = append
    post_config["*"].output_channel.unbuffered = unbuffered

    crp.log_config = config
    crp.log_post_config = post_config
    crp.compress_log_file = compress_after_run
    crp.default_log_channel = "output-channel"


def crp_set_corpus(crp, corpus):
    """
    :param CommonRasrParameters crp:
    :param meta.CorpusObject corpus: object with corpus_file, audio_dir, audio_format, duration
    """
    config = RasrConfig()
    config.file = corpus.corpus_file
    config.audio_dir = corpus.audio_dir
    config.warn_about_unexpected_elements = True
    config.capitalize_transcriptions = False
    config.progress_indication = "global"

    crp.corpus_config = config
    crp.audio_format = corpus.audio_format
    crp.corpus_duration = corpus.duration
