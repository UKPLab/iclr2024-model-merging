__all__ = ["DownloadJob"]

import subprocess as sp

from sisyphus import *

from i6_core.util import check_file_sha256_checksum

Path = setup_path(__package__)


class DownloadJob(Job):
    """
    Download an arbitrary file with optional checksum verification

    If a checksum is provided the url will not be hashed
    """

    def __init__(self, url, target_filename=None, checksum=None):
        """

        :param str url:
        :param str|None target_filename: explicit output filename, if None tries to detect the filename from the url
        :param str|None checksum: A sha256 checksum to verify the file
        """
        self.url = url
        self.target_filename = (
            target_filename if target_filename else url.split("/")[-1]
        )
        self.checksum = checksum

        self.out_file = self.output_path(self.target_filename)

        if self.target_filename:
            self.set_vis_name(target_filename)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        sp.check_call(["wget", "-O", self.out_file.get_path(), self.url])
        if self.checksum:
            check_file_sha256_checksum(self.out_file.get_path(), self.checksum)

    @classmethod
    def hash(cls, parsed_args):
        if parsed_args["checksum"] is not None:
            parsed_args.pop("url")
        return super().hash(parsed_args)
