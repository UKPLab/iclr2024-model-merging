__all__ = ["MakeJob"]

import logging
import os
import subprocess as sp
import shutil
from tempfile import TemporaryDirectory
from typing import Iterator, Optional, List, Dict

from sisyphus import tk, gs, setup_path, Job, Task

Path = setup_path(__package__)


class MakeJob(Job):
    """
    Executes a sequence of make commands in a given folder
    """

    def __init__(
        self,
        folder: tk.Path,
        make_sequence: Optional[List[str]] = None,
        configure_opts: Optional[List[str]] = None,
        num_processes: int = 1,
        output_folder_name: Optional[str] = "repository",
        link_outputs: Optional[Dict[str, str]] = None,
    ):
        """

        :param folder: folder in which the make commands are executed,
            e.g. a GitCloneRepositoryJob output
        :param make_sequence: list of options that are given to the make calls.
            defaults to ["all"] i.e. "make all" is executed
        :param configure_opts: if given, runs ./configure with these options before make
        :param num_processes: number of parallel running make processes
        :param output_folder_name: name of the output path folder, if None,
            the repo is not copied as output
        :param link_outputs: provide "output_name": "local/repo/file_folder" pairs to
            link (or copy if output_folder_name=None) files or directories as output.
            This can be used to access single binaries or a binary folder instead of the whole repository.
        """
        self.folder = folder
        self.make_sequence = make_sequence if make_sequence is not None else ["all"]
        self.configure_opts = configure_opts
        self.num_processes = num_processes
        self.output_folder_name = output_folder_name
        self.link_outputs = link_outputs

        self.rqmt = {"cpu": num_processes, "mem": 4}

        assert (
            output_folder_name or link_outputs
        ), "please provide either output_folder_name or link_outputs, otherwise the output will be empty"

        if output_folder_name:
            self.out_repository = self.output_path(output_folder_name)

        if link_outputs:
            self.out_links = {}
            for key in link_outputs.keys():
                self.out_links[key] = self.output_path(key)

    def tasks(self) -> Iterator[Task]:
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        with TemporaryDirectory(prefix=gs.TMP_PREFIX) as temp_dir:
            try:
                shutil.rmtree(temp_dir)
                shutil.copytree(self.folder.get_path(), temp_dir, symlinks=True)

                if self.configure_opts is not None:
                    args = ["./configure"] + self.configure_opts
                    logging.info("running command: %s" % " ".join(args))
                    sp.run(args, cwd=temp_dir, check=True)

                for command in self.make_sequence:
                    args = ["make"]
                    args.extend(command.split())
                    if "-j" not in args:
                        args.extend(["-j", f"{self.num_processes}"])

                    logging.info("running command: %s" % " ".join(args))
                    sp.run(args, cwd=temp_dir, check=True)

                if self.output_folder_name:
                    shutil.copytree(
                        temp_dir, self.out_repository.get_path(), symlinks=True
                    )
                if self.link_outputs:
                    for key, path in self.link_outputs.items():
                        trg = self.out_links[key].get_path()
                        if self.output_folder_name:
                            src = os.path.join(self.out_repository.get_path(), path)
                            os.symlink(src, trg)
                        else:
                            src = os.path.join(temp_dir, path)
                            if os.path.isdir(src):
                                shutil.rmtree(trg, ignore_errors=True)
                                shutil.copytree(src, trg)
                            else:
                                shutil.copy(src, trg)
            except Exception as e:
                shutil.copytree(temp_dir, "crash_repo")
                raise e

    @classmethod
    def hash(cls, kwargs):
        d = kwargs.copy()
        d.pop("num_processes")
        return super().hash(d)
