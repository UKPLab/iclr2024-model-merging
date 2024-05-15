__all__ = ["BlissToOggZipJob"]

import os
import shutil
import subprocess as sp
import tempfile
import zipfile

from i6_core.util import MultiOutputPath, relink

from sisyphus import *

Path = setup_path(__package__)


class BlissToOggZipJob(Job):
    """
    This Job is a wrapper around the RETURNN tool bliss-to-ogg-zip.py.

    """

    __sis_hash_exclude__ = {"no_audio": False, "ffmpeg_acodec": None}

    def __init__(
        self,
        bliss_corpus,
        segments=None,
        rasr_cache=None,
        raw_sample_rate=None,
        feat_sample_rate=None,
        no_conversion=False,
        no_audio=False,
        ffmpeg_acodec=None,
        returnn_python_exe=None,
        returnn_root=None,
    ):
        """
        use RETURNN to dump data into an ogg zip file

        The output zip archive can be created in a single run or in a parallel way using a MultiOutputPath for segments,
        e.g.
        > segments = SegmentCorpusJob(corpus, concurrent).out_segment_path
        > ogg_zip_job = BlissToOggZipJob(corpus, segments=segments)

        :param str|Path bliss_corpus: bliss corpus file
        :param str|Path|MultiOutputPath segments: RASR segment file
            If a MultiOutputPath object is given, a single zip archive is created for each segment list split and all
            archives are merged at the end.
        :param str|Path rasr_cache: feature rasr cache
        :param int raw_sample_rate: raw audio sampling rate
        :param int feat_sample_rate: feature sampling rate
        :param bool no_conversion: do not call the actual conversion, assume the audio files are already correct
        :param bool no_audio: do not add audio files
        :param str ffmpeg_acodec: force audio codec for ffmpeg calls
        :param Path|str returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param Path|str returnn_root: file path to the RETURNN repository root folder
        """
        self.bliss_corpus = bliss_corpus
        self.segments = segments
        self.rasr_cache = rasr_cache
        self.raw_sample_rate = raw_sample_rate
        self.feat_sample_rate = feat_sample_rate
        self.no_conversion = no_conversion
        self.no_audio = no_audio
        self.ffmpeg_acodec = ffmpeg_acodec
        self.concurrent = (
            len(segments.hidden_paths) if isinstance(segments, MultiOutputPath) else 1
        )

        self.returnn_python_exe = (
            returnn_python_exe
            if returnn_python_exe is not None
            else gs.RETURNN_PYTHON_EXE
        )
        self.returnn_root = (
            returnn_root if returnn_root is not None else gs.RETURNN_ROOT
        )

        self.zip_subarchives = (
            MultiOutputPath(
                self,
                "ogg.$(TASK).zip",
                {i: "ogg.{}.zip".format(i) for i in range(1, self.concurrent + 1)},
                cached=True,
            )
            if self.concurrent > 1
            else None
        )

        self.out_ogg_zip = self.output_path("out.ogg.zip")

        self.rqmt = None
        self.merge_rqmt = {"cpu": 1, "mem": 1, "time": 1}

    def tasks(self):
        if self.rqmt:
            yield Task("run", rqmt=self.rqmt, args=range(1, self.concurrent + 1))
        else:
            yield Task("run", mini_task=True, args=range(1, self.concurrent + 1))
        if self.concurrent > 1:
            if self.merge_rqmt:
                yield Task("merge", rqmt=self.merge_rqmt)
            else:
                yield Task("merge", mini_task=True)

    def run(self, task_id):
        output = (
            self.zip_subarchives.hidden_paths[task_id]
            if self.concurrent > 1
            else "out.ogg.zip"
        )
        args = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(
                tk.uncached_path(self.returnn_root), "tools/bliss-to-ogg-zip.py"
            ),
            tk.uncached_path(self.bliss_corpus),
            "--output",
            output,
        ]
        if isinstance(self.segments, MultiOutputPath):
            args.extend(["--subset_segment_file", self.segments.hidden_paths[task_id]])
        elif self.segments is not None:
            args.extend(["--subset_segment_file", tk.uncached_path(self.segments)])

        if self.no_audio:
            args.extend(["--no_ogg"])
        elif self.no_conversion:
            args.extend(["--no_conversion"])
        elif self.ffmpeg_acodec:
            args.extend(["--ffmpeg_acodec", self.ffmpeg_acodec])
        else:
            if self.rasr_cache is not None:
                args.extend(["--sprint_cache", tk.uncached_path(self.rasr_cache)])
            if self.raw_sample_rate is not None:
                args.extend(["--raw_sample_rate", str(self.raw_sample_rate)])
            if self.feat_sample_rate is not None:
                args.extend(["--feat_sample_rate", str(self.feat_sample_rate)])

        sp.check_call(args)
        if self.concurrent == 1:
            relink("out.ogg.zip", self.out_ogg_zip.get_path())

    def merge(self):
        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp_dir:
            # extract all subarchives
            for zip_subarchive in self.zip_subarchives.hidden_paths.values():
                with zipfile.ZipFile(
                    zip_subarchive, mode="r", compression=zipfile.ZIP_DEFLATED
                ) as zip_file:
                    zip_file.extractall(tmp_dir)
                os.remove(zip_subarchive)

            # create output folder
            assert self.out_ogg_zip.get().endswith(".zip")
            output_folder = os.path.join(
                tmp_dir, os.path.basename(self.out_ogg_zip.get())[: -len(".zip")]
            )
            os.mkdir(output_folder)

            # merge meta files, remove from subarchives
            meta_file = output_folder + ".txt"
            with open(meta_file, "w") as mf:
                mf.write("[\n")
                for zip_subarchive in self.zip_subarchives.hidden_paths.values():
                    sub_meta_file = os.path.join(
                        tmp_dir, zip_subarchive.replace(".zip", ".txt")
                    )
                    with open(sub_meta_file, "r") as smf:
                        for line in smf.readlines():
                            if not (line.startswith("[") or line.startswith("]")):
                                mf.write(line)
                    os.remove(sub_meta_file)
                mf.write("]\n")

            # move all folders to output folder
            cmd = [
                "rsync",
                "-av",
                os.path.join(
                    tmp_dir,
                    os.path.basename(self.zip_subarchives.path_template).replace(
                        ".$(TASK).zip", ".*/*"
                    ),
                ),
                output_folder,
            ]
            print("$ {}".format(" ".join(cmd)))
            sp.check_call(" ".join(cmd), shell=True)

            # remove extracted folders
            for zip_subarchive in self.zip_subarchives.hidden_paths.values():
                shutil.rmtree(os.path.join(tmp_dir, zip_subarchive[: -len(".zip")]))

            # compress output folder to zip archive
            with zipfile.ZipFile(
                os.path.join(tmp_dir, os.path.basename(self.out_ogg_zip.get())),
                mode="a",
                compression=zipfile.ZIP_DEFLATED,
            ) as zip_file:
                for dirpath, dirnames, filenames in os.walk(tmp_dir):
                    for name in sorted(dirnames + filenames):
                        if name.endswith(".zip"):
                            continue
                        path = "{}/{}".format(dirpath, name)
                        assert path.startswith(tmp_dir + "/")
                        zip_path = path[len(tmp_dir) + 1 :]
                        print("Adding:", zip_path)
                        zip_file.write(path, zip_path)

            shutil.move(zip_file.filename, self.out_ogg_zip.get())

    @classmethod
    def hash(cls, parsed_args):
        del parsed_args["returnn_python_exe"]
        return super().hash(parsed_args)
