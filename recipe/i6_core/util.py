import gzip
import os
import shutil
import stat
import subprocess as sp
import xml.dom.minidom
import xml.etree.ElementTree as ET

from sisyphus import *
from sisyphus.delayed_ops import DelayedBase

Path = setup_path(__package__)
Variable = tk.Variable


class MultiPath:
    def __init__(
        self,
        path_template,
        hidden_paths,
        cached=False,
        path_root=None,
        hash_overwrite=None,
    ):
        self.path_template = path_template
        self.hidden_paths = hidden_paths
        self.cached = cached
        self.path_root = path_root
        self.hash_overwrite = hash_overwrite

    def __str__(self):
        if self.path_root is not None:
            result = os.path.join(self.path_root, self.path_template)
        else:
            result = self.path_template
        if self.cached:
            result = gs.file_caching(result)
        return result

    def __sis_state__(self):
        return {
            "path_template": self.path_template
            if self.hash_overwrite is None
            else self.hash_overwrite,
            "hidden_paths": self.hidden_paths,
            "cached": self.cached,
        }


class MultiOutputPath(MultiPath):
    def __init__(self, creator, path_template, hidden_paths, cached=False):
        super().__init__(
            os.path.join(creator._sis_path(gs.JOB_OUTPUT), path_template),
            hidden_paths,
            cached,
            gs.BASE_DIR,
        )


def write_paths_to_file(file, paths):
    with open(tk.uncached_path(file), "w") as f:
        for p in paths:
            f.write(tk.uncached_path(p) + "\n")


def zmove(src, target):
    src = tk.uncached_path(src)
    target = tk.uncached_path(target)

    if not src.endswith(".gz"):
        tmp_path = src + ".gz"
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        sp.check_call(["gzip", src])
        src += ".gz"
    if not target.endswith(".gz"):
        target += ".gz"

    shutil.move(src, target)


def delete_if_exists(file):
    if os.path.exists(file):
        os.remove(file)


def delete_if_zero(file):
    if os.path.exists(file) and os.stat(file).st_size == 0:
        os.remove(file)


def backup_if_exists(file):
    if os.path.exists(file):
        dir, base = os.path.split(file)
        base = add_suffix(base, ".gz")
        idx = 1
        while os.path.exists(os.path.join(dir, "backup.%.4d.%s" % (idx, base))):
            idx += 1
        zmove(file, os.path.join(dir, "backup.%.4d.%s" % (idx, base)))


def remove_suffix(string, suffix):
    if string.endswith(suffix):
        return string[: -len(suffix)]
    return string


def add_suffix(string, suffix):
    if not string.endswith(suffix):
        return string + suffix
    return string


def partition_into_tree(l, m):
    """Transforms the list l into a nested list where each sub-list has at most length m + 1"""
    nextPartition = partition = l
    while len(nextPartition) > 1:
        partition = nextPartition
        nextPartition = []
        d = len(partition) // m
        mod = len(partition) % m
        if mod <= d:
            p = 0
            for i in range(mod):
                nextPartition.append(partition[p : p + m + 1])
                p += m + 1
            for i in range(d - mod):
                nextPartition.append(partition[p : p + m])
                p += m
            assert p == len(partition)
        else:
            p = 0
            for i in range(d):
                nextPartition.append(partition[p : p + m])
                p += m
            nextPartition.append(partition[p : p + mod])
            assert p + mod == len(partition)
    return partition


def reduce_tree(func, tree):
    return func([(reduce_tree(func, e) if type(e) == list else e) for e in tree])


def uopen(path, *args, **kwargs):
    path = tk.uncached_path(path)
    if path.endswith(".gz"):
        return gzip.open(path, *args, **kwargs)
    else:
        return open(path, *args, **kwargs)


def get_val(var):
    if isinstance(var, Variable):
        return var.get()
    return var


def num_cart_labels(path):
    path = tk.uncached_path(path)
    if path.endswith(".gz"):
        open_func = gzip.open
    else:
        open_func = open
    file = open_func(path, "rt")
    tree = ET.parse(file)
    file.close()
    all_nodes = tree.findall("binary-tree//node")
    return len([n for n in all_nodes if n.find("node") is None])


def chunks(l, n):
    """
    :param list[T] l: list which should be split into chunks
    :param int n: number of chunks
    :return: yields n chunks
    :rtype: list[list[T]]
    """
    bigger_count = len(l) % n
    start = 0
    block_size = len(l) // n
    for i in range(n):
        end = start + block_size + (1 if i < bigger_count else 0)
        yield l[start:end]
        start = end


def relink(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    os.link(src, dst)


def cached_path(path):
    if tk.is_path(path) and path.cached:
        caching_command = gs.file_caching(tk.uncached_path(path))
        caching_command = caching_command.replace("`", "")
        caching_command = caching_command.split(" ")
        if len(caching_command) > 1:
            ret = sp.check_output(caching_command)
            return ret.strip()
    return tk.uncached_path(path)


def write_xml(filename, element_tree, prettify=True):
    """
    writes element tree to xml file
    :param Union[Path, str] filename: name of desired output file
    :param ET.ElementTree|ET.Element element_tree: element tree which should be written to file
    :param bool prettify: prettify the xml. Warning: be careful with this option if you care about whitespace in the xml.
    """

    def remove_unwanted_whitespace(elem):
        import re

        has_non_whitespace = re.compile(r"\S")
        for element in elem.iter():
            if not re.search(has_non_whitespace, str(element.tail)):
                element.tail = ""
            if not re.search(has_non_whitespace, str(element.text)):
                element.text = ""

    if isinstance(element_tree, ET.ElementTree):
        root = element_tree.getroot()
    elif isinstance(element_tree, ET.Element):
        root = element_tree
    else:
        assert False, "please provide an ElementTree or Element"

    if prettify:
        remove_unwanted_whitespace(root)
        xml_string = xml.dom.minidom.parseString(ET.tostring(root)).toprettyxml(
            indent=" " * 2
        )
    else:
        xml_string = ET.tostring(root, encoding="unicode")

    with uopen(filename, "wt") as f:
        f.write(xml_string)


def create_executable(filename, command):
    """
    create an executable .sh file calling a single command
    :param str filename: executable name ending with .sh
    :param list[str] command: list representing the command and parameters
    :return:
    """
    assert filename.endswith(".sh")
    with open(filename, "wt") as f:
        f.write("#!/usr/bin/env bash\n%s" % " ".join(command))
    os.chmod(
        filename,
        stat.S_IRUSR
        | stat.S_IRGRP
        | stat.S_IROTH
        | stat.S_IWUSR
        | stat.S_IXUSR
        | stat.S_IXGRP
        | stat.S_IXOTH,
    )


def compute_file_sha256_checksum(filename):
    """
    Computes the sha256sum for a file

    :param str filename: a single file to be checked
    :return: checksum
    :rtype:str
    """
    checksum_command_output = sp.check_output(["sha256sum", filename])
    return checksum_command_output.decode().strip().split(" ")[0]


def check_file_sha256_checksum(filename, reference_checksum):
    """
    Validates the sha256sum for a file against the target checksum

    :param str filename: a single file to be checked
    """
    assert compute_file_sha256_checksum(filename) == reference_checksum


def instanciate_delayed(o):
    """
    Recursively traverses a structure and calls .get() on all
    existing Delayed Operations, especially Variables in the structure

    :param Any o: nested structure that may contain DelayedBase objects
    :return:
    """
    if isinstance(o, DelayedBase):
        o = o.get()
    elif isinstance(o, list):
        for k in range(len(o)):
            o[k] = instanciate_delayed(o[k])
    elif isinstance(o, tuple):
        o = tuple(instanciate_delayed(e) for e in o)
    elif isinstance(o, dict):
        for k in o:
            o[k] = instanciate_delayed(o[k])
    return o
