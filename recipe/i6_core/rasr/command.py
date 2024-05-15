__all__ = ["RasrCommand"]

import logging
import time

from sisyphus import *

Path = setup_path(__package__)

from i6_core.util import *


class RasrCommand:
    """
    Mixin for :class:`Job`.
    """

    RETRY_WAIT_TIME = 5.0
    NO_RETRY_AFTER_TIME = 10.0 * 60.0  # do not retry job after it ran for 10 minutes

    def log_file_output_path(self, name, crp, parallel):
        """
        :param str name:
        :param rasr.crp.CommonRasrParameters crp:
        :param int|bool parallel:
        :rtype: Path
        """
        suffix = ".gz" if crp.compress_log_file else ""
        if parallel:
            num_logs = parallel if type(parallel) == int else crp.concurrent
            return dict(
                (task_id, self.output_path("%s.log.%d%s" % (name, task_id, suffix)))
                for task_id in range(1, num_logs + 1)
            )
        return self.output_path("%s.log%s" % (name, suffix))

    @staticmethod
    def write_config(config, post_config, filename):
        """
        :param rasr.RasrConfig config:
        :param rasr.RasrConfig post_config:
        :param str filename:
        """
        config._update(post_config)
        with open(filename, "wt") as f:
            f.write(repr(config))

    @staticmethod
    def write_run_script(exe, config, filename="run.sh", extra_code="", extra_args=""):
        """
        :param str exe:
        :param str config:
        :param str filename:
        :param str extra_code:
        """
        with open(filename, "wt") as f:
            f.write(
                """\
#!/usr/bin/env bash
set -ueo pipefail

if [[ $# -gt 0 ]]; then
  TASK=$1;
  shift;
else
  echo "No TASK-id given";
  exit 1;
fi

if [ $# -gt 0 ]; then
  LOGFILE=$1;
  shift;
else
  LOGFILE=rasr.log
fi

%s

%s --config=%s --*.TASK=$TASK --*.LOGFILE=$LOGFILE %s $@\
              """
                % (extra_code, exe, config, extra_args)
            )
            os.chmod(filename, 0o755)

    @classmethod
    def get_rasr_exe(cls, exe_name, rasr_root, rasr_arch):
        """
        :param str exe_name:
        :param str rasr_root:
        :param str rasr_arch:
        :return: path to a rasr binary with the default path pattern inside the repsoitory
        :rtype: str
        """
        exe = os.path.join(
            rasr_root, "arch", rasr_arch, "%s.%s" % (exe_name, rasr_arch)
        )
        return exe

    @classmethod
    def default_exe(cls, exe_name):
        """
        Extract executable path from the global sisyphus settings

        :param str exe_name:
        :rtype: str
        """
        return cls.get_rasr_exe(exe_name, gs.RASR_ROOT, gs.RASR_ARCH)

    @classmethod
    def select_exe(cls, specific_exe, default_exe_name):
        """
        :param str|None specific_exe:
        :param str default_exe_name:
        :return: path to exe
        :rtype: str
        """
        if specific_exe is None:
            return cls.default_exe(default_exe_name)
        return specific_exe

    def run_script(self, task_id, log_file, cmd="./run.sh", args=None, retries=2):
        args = [] if args is None else args
        tmp_log_file = remove_suffix(
            os.path.basename(tk.uncached_path(log_file)), ".gz"
        )
        self.run_cmd(cmd, [task_id, tmp_log_file] + args, retries)
        zmove(tmp_log_file, tk.uncached_path(log_file))

    def run_cmd(self, cmd, args=None, retries=2):
        """
        :param str cmd:
        :param list[str]|None args:
        :param int retries:
        """
        args = [] if args is None else args
        retries = max(0, retries)

        start_time = time.monotonic()

        for t in range(retries + 1):
            try:
                self.cleanup_before_run(cmd, t, *args)
                sp.check_call([cmd] + [str(arg) for arg in args])
                break
            except sp.CalledProcessError as e:
                logging.warning(
                    "cmd %s (args: %s) failed with exit code %d"
                    % (cmd, str(args), e.returncode)
                )
                elapsed = time.monotonic() - start_time
                if t == retries or elapsed >= self.NO_RETRY_AFTER_TIME:
                    raise
                time.sleep(self.RETRY_WAIT_TIME)

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        pass
