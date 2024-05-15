__all__ = [
    "PipelineJob",
    "ConcatenateJob",
    "HeadJob",
    "TailJob",
    "SetDifferenceJob",
    "WriteToTextFileJob",
]

import os
from collections.abc import Iterable

from sisyphus import Job, Task, Path, global_settings as gs
from sisyphus.delayed_ops import DelayedBase

import i6_core.util as util


class PipelineJob(Job):
    """
    Reads a text file and applies a list of piped shell commands
    """

    def __init__(
        self,
        text_files,
        pipeline,
        zip_output=False,
        check_equal_length=False,
        mini_task=False,
    ):
        """
        :param iterable[Path]|Path text_files: text file (raw or gz) or list of files to be processed
        :param list[str|DelayedBase] pipeline: list of shell commands to form the pipeline,
            can be empty to use the job for concatenation or gzip compression only.
        :param bool zip_output: apply gzip to the output
        :param bool check_equal_length: the line count of the input and output should match
        :param bool mini_task: the pipeline should be run as mini_task
        """
        assert text_files is not None
        self.text_files = text_files
        self.pipeline = pipeline
        self.zip_output = zip_output
        self.check_equal_length = check_equal_length
        self.mini_task = mini_task

        if zip_output:
            self.out = self.output_path("out.gz")
        else:
            self.out = self.output_path("out")

        self.rqmt = None

    def tasks(self):
        if not self.rqmt:
            # estimate rqmt if not set explicitly
            if isinstance(self.text_files, (list, tuple)):
                size = sum(text.estimate_text_size() / 1024 for text in self.text_files)
            else:
                size = self.text_files.estimate_text_size() / 1024

            if size <= 128:
                time = 2
                mem = 2
            elif size <= 512:
                time = 3
                mem = 3
            elif size <= 1024:
                time = 4
                mem = 3
            elif size <= 2048:
                time = 6
                mem = 4
            else:
                time = 8
                mem = 4
            cpu = 1
            self.rqmt = {"time": time, "mem": mem, "cpu": cpu}

        if self.mini_task:
            yield Task("run", mini_task=True)
        else:
            yield Task("run", rqmt=self.rqmt)

    def run(self):
        pipeline = self.pipeline.copy()
        if self.zip_output:
            pipeline.append("gzip")
        pipe = " | ".join([str(i) for i in pipeline])
        if isinstance(self.text_files, (list, tuple)):
            inputs = " ".join(i.get_cached_path() for i in self.text_files)
        else:
            inputs = self.text_files.get_cached_path()
        if pipe:
            self.sh("zcat -f %s | %s > %s" % (inputs, pipe, self.out.get_path()))
        else:
            self.sh("zcat -f %s > %s" % (inputs, self.out.get_path()))

        # assume that we do not want empty pipe results
        assert not (os.stat(str(self.out)).st_size == 0), "Pipe result was empty"

        input_length = int(self.sh("zcat -f %s | sed '$a\\' | wc -l" % inputs, True))
        assert input_length > 0
        output_length = int(self.sh("zcat -f %s | wc -l" % self.out.get_path(), True))
        assert output_length > 0
        if self.check_equal_length:
            assert (
                input_length == output_length
            ), "pipe input and output lengths do not match"

    @classmethod
    def hash(cls, parsed_args):
        args = parsed_args.copy()
        del args["check_equal_length"]
        del args["mini_task"]
        return super(PipelineJob, cls).hash(args)


class ConcatenateJob(Job):
    """
    Concatenate all given input files (gz or raw)
    """

    __sis_hash_exclude = {"zip_out": True, "out_name": "out"}

    def __init__(self, text_files, zip_out=True, out_name="out"):
        """
        :param list[Path] text_files: input text files
        :param bool zip_out: apply gzip to the output
        :param str out_name: user specific name
        """
        assert text_files

        # ensure sets are always merged in the same order
        if isinstance(text_files, set):
            text_files = list(text_files)
            text_files.sort(key=lambda x: str(x))

        assert isinstance(text_files, list)

        # Skip this job if only one input is present
        if len(text_files) == 1:
            self.out = text_files.pop()
        else:
            if zip_out:
                self.out = self.output_path(out_name + ".gz")
            else:
                self.out = self.output_path(out_name)

        for input in text_files:
            assert isinstance(input, Path) or isinstance(
                input, str
            ), "input to Concatenate is not a valid path"

        self.text_files = text_files
        self.zip_out = zip_out

    def tasks(self):
        yield Task("run", rqmt={"mem": 3, "time": 3})

    def run(self):
        self.f_list = " ".join(gs.file_caching(str(i)) for i in self.text_files)
        if self.zip_out:
            self.sh("zcat -f {f_list} | gzip > {out}")
        else:
            self.sh("zcat -f {f_list} > {out}")


class HeadJob(Job):
    """
    Return the head of a text file, either absolute or as ratio (provide one)
    """

    __sis_hash_exclude__ = {"zip_output": True}

    def __init__(self, text_file, num_lines=None, ratio=None, zip_output=True):
        """
        :param Path text_file: text file (gz or raw)
        :param int num_lines: number of lines to extract
        :param float ratio: ratio of lines to extract
        """
        assert num_lines or ratio, "please specify either lines or ratio"
        assert not (num_lines and ratio), "please specify only lines or ratio, not both"
        if ratio:
            assert ratio <= 1

        self.text_file = text_file
        self.num_lines = num_lines
        self.ratio = ratio
        self.zip_output = zip_output

        self.out = self.output_path("out.gz")
        self.length = self.output_var("length")

    def tasks(self):
        yield Task(
            "run",
            rqmt={
                "cpu": 1,
                "time": 2,
                "mem": 4,
            },
        )

    def run(self):
        if self.ratio:
            assert not self.num_lines
            length = int(self.sh("zcat -f {text_file} | wc -l", True))
            self.lines = int(length * self.ratio)

        pipeline = "zcat -f {text_file} | head -n {num_lines}"
        if self.zip_output:
            pipeline += " | gzip"
        pipeline += " > {out}"

        self.sh(
            pipeline,
            except_return_codes=(141,),
        )
        self.length.set(self.num_lines)


class TailJob(HeadJob):
    """
    Return the tail of a text file, either absolute or as ratio (provide one)
    """

    def run(self):
        if self.ratio:
            assert not self.lines
            length = int(self.sh("zcat -f {text_file} | wc -l", True))
            self.lines = int(length * self.ratio)

        self.sh("zcat -f {text_file} | tail -n {num_lines} | gzip > {out}")


class SetDifferenceJob(Job):
    """
    Return the set difference of two text files, where one line is one element.
    """

    def __init__(self, minuend, subtrahend, gzipped=False):
        """
        This job performs the set difference minuend - subtrahend. Unlike the bash utility comm, the two files
        do not need to be sorted.
        :param Path minuend: left-hand side of the set subtraction
        :param Path subtrahend: right-hand side of the set subtraction
        :param bool gzipped: whether the output should be compressed in gzip format
        """
        self.minuend = minuend
        self.subtrahend = subtrahend

        outfile_ext = "txt.gz" if gzipped else "txt"
        self.out_file = self.output_path("diff.%s" % outfile_ext)

        self.rqmt = {"cpu": 1, "time": 1, "mem": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        with util.uopen(self.minuend, "rt") as fin:
            file_set1 = set(fin.read().split("\n"))
        with util.uopen(self.subtrahend, "rt") as fin:
            file_set2 = set(fin.read().split("\n"))
        with util.uopen(self.out_file, "wt") as fout:
            fout.write("\n".join(sorted(file_set1.difference(file_set2))))


class WriteToTextFileJob(Job):
    """
    Write a given content into a text file, one entry per line
    """

    def __init__(
        self,
        content,
    ):
        """
        :param list|dict|str content: input which will be written into a text file
        """
        self.content = content

        self.out_file = self.output_path("file.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with open(self.out_file.get_path(), "w") as f:
            if isinstance(self.content, str):
                f.write(self.content)
            elif isinstance(self.content, dict):
                for key, val in self.content.items():
                    f.write(f"{key}: {val}\n")
            elif isinstance(self.content, Iterable):
                for line in self.content:
                    f.write(f"{line}\n")
            else:
                raise NotImplementedError
