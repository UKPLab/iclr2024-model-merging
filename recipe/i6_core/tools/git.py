__all__ = ["CloneGitRepositoryJob"]

import logging
import subprocess as sp

from sisyphus import *

Path = setup_path(__package__)


class CloneGitRepositoryJob(Job):
    """
    Clone a git repository given optional branch name and commit hash
    """

    def __init__(
        self, url, branch=None, commit=None, checkout_folder_name="repository"
    ):
        """

        :param str url: git repository url
        :param str branch: git branch name
        :param str commit: git commit hash
        :param str checkout_folder_name: name of the output path repository folder
        """
        self.url = url
        self.branch = branch
        self.commit = commit

        self.out_repository = self.output_path(checkout_folder_name, True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        args = ["git", "clone", self.url]
        if self.branch is not None:
            args.extend(["-b", self.branch])
        repository_dir = self.out_repository.get_path()
        args += [repository_dir]
        logging.info("running command: %s" % " ".join(args))
        sp.run(args, check=True)
        if self.commit is not None:
            args = ["git", "checkout", self.commit]
            logging.info("running command: %s" % " ".join(args))
            sp.run(args, cwd=repository_dir, check=True)
