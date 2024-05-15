__all__ = [
    "PrintTableJob",
    "TableReport",
    "ScliteSummaryJob",
    "ScliteLurSummaryJob",
    "KaldiSummaryJob",
]

import os
import re

from sisyphus import *

Path = setup_path(__package__)
Variable = tk.Variable


def write_table(f, data, header, row_names, col_names):
    """
    :param f: Buffer for writing the string. Examples: io.StringIO, io.TextIOWrapper
    :param data (dict): contains strings at keys data[(col, row)]
    :param header (str): header of the first column
    :param col_names ([str]): list of columns in order of appearance
    :param row_names([str]): list of rows in order of appearance
    :return: None
    """
    first_col_width = max(len(header), max(len(r) for r in row_names))
    col_widths = [
        max(len(c), max(len(data.get((c, r), "")) for r in row_names))
        for c in col_names
    ]

    f.write("%*s" % (first_col_width, header))
    for w, c in zip(col_widths, col_names):
        f.write(" | %*s" % (w, c))
    f.write("\n")

    f.write("-" * first_col_width)
    for w in col_widths:
        f.write("-+-" + "-" * w)
    f.write("\n")

    for r in row_names:
        f.write("%*s" % (first_col_width, r))
        for w, c in zip(col_widths, col_names):
            if (c, r) in data:
                f.write(" | %*s" % (w, data[(c, r)]))
            else:
                f.write(" | %*s" % (w, " "))
        f.write("\n")


def natural_keys(text):
    """
    https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    """

    def atoi(text):
        return int(text) if text.isdigit() else text

    return [atoi(c) for c in re.split(r"(\d+)", text)]


class PrintTableJob(Job):
    def __init__(
        self,
        data,
        header,
        col_names=None,
        row_names=None,
        sort_cols=False,
        sort_rows=False,
        precision=2,
    ):
        """
        :param data (dict): contains strings at keys data[(col, row)]
        :param header (str): header of the first column
        :param col_names ([str]): list of columns in order of appearance
        :param row_names([str]): list of rows in order of appearance
        :param sort_rows(bool): if true, the rows will be sorted alphanumerically
        :param sort_cols(bool): if true, the columns will be sorted alphanumerically
        """
        self.data = data
        self.header = header
        self.precision = precision
        self.col_names = (
            list(set([x[0] for x in data])) if col_names is None else col_names
        )
        self.row_names = (
            list(set([x[1] for x in data])) if row_names is None else row_names
        )
        if sort_cols:
            self.col_names.sort(key=natural_keys)
        if sort_rows:
            self.row_names.sort(key=natural_keys)

        self.summary = self.output_path("table.txt")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        data = {}
        for (col, row), var in self.data.items():
            if isinstance(var, Variable):
                v = var.get()
            else:
                v = var
            if isinstance(v, float):
                v = str(round(v, self.precision))
            else:
                v = str(v)
            data[(col, row)] = v

        with open(self.summary.get_path(), "wt") as f:
            write_table(
                f=f,
                data=data,
                header=self.header,
                row_names=self.row_names,
                col_names=self.col_names,
            )


class TableReport:
    def __init__(self, header, precision=2):
        self.header = header
        self.precision = precision
        self.data = {}
        self.col_names = []
        self.row_names = []

    def add_entry(self, col, row, var):
        self.data[(col, row)] = var
        self.update_names()

    def update_names(self):
        self.col_names = list(set([x[0] for x in self.data]))
        self.row_names = list(set([x[1] for x in self.data]))
        self.col_names.sort(key=natural_keys)
        self.row_names.sort(key=natural_keys)

    def __call__(self):
        data = {}
        for (col, row), var in self.data.items():
            if isinstance(var, Variable):
                if var.available():
                    v = var.get()
                else:
                    v = " "
            else:
                v = var
            if isinstance(v, float):
                v = str(round(v, self.precision))
            else:
                v = str(v)
            data[(col, row)] = v

        from io import StringIO

        f = StringIO()
        write_table(
            f=f,
            data=data,
            header=self.header,
            row_names=self.row_names,
            col_names=self.col_names,
        )
        return f.getvalue()


class ScliteSummaryJob(PrintTableJob):
    def __init__(
        self,
        data,
        header,
        col_names=None,
        row_names=None,
        file_name="sclite.dtl",
        **kwargs,
    ):
        super().__init__(data, header, col_names, row_names, **kwargs)
        self.file_name = file_name

        self.summary = self.output_path("summary.txt")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        data = {}
        for report_dir, col, row in self.data:
            sclite_file = os.path.join(tk.uncached_path(report_dir), self.file_name)
            data[(col, row)] = self.wer(sclite_file)
        self.data = data
        super().run()

    @staticmethod
    def wer(path):
        regex = re.compile("^Percent Total Error *= *(\\d+\.\\d%).*")
        with open(path, "rt") as f:
            for line in f:
                m = regex.match(line)
                if m is not None:
                    return m.group(1)
        return None


class ScliteLurSummaryJob(Job):
    def __init__(self, data, file_name="sclite.lur"):
        """
        Prints a table containing all sclite lur results
        :param data: {name:str , report_dir:str}
        """
        wer_pattern = "\[\S*\]\s*(\S*)"
        self.prog = re.compile(wer_pattern)
        self.data = data
        self.summary = self.output_path("summary.txt")
        self.file_name = file_name

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        dicts = []
        for data in self.data:
            name = data["name"]
            report_dir = data["report_dir"]
            sclite_file = os.path.join(tk.uncached_path(report_dir), self.file_name)
            dicts.append({"name": name, "data": self.parse_lur(sclite_file)})
        self.dict2table(dicts)

    def parse_lur(self, file_path):
        with open(file_path, "r") as infile:
            for line in infile:
                if "SPKR" in line:
                    groups = line.replace(" ", "").split("|")
                    groups = [x for x in groups if x.isalnum()]
                    groups.remove("SPKR")
                    continue
                if "Mean" in line:
                    res = self.prog.findall(line)
                    break
            return dict(zip(groups, res))

    def dict2table(self, dicts):
        """
        Gets a list of dictionarys and creates a table
        :param dicts: [{name : str , data : {col:float, col:float....} }, ... ]
        :return:
        """
        rows = []
        cols = []
        for dict in dicts:
            rows.append(dict["name"])
            for key in dict["data"]:
                if key not in cols:
                    cols.append(key)

        # check that all rows are unique
        assert len(rows) == len(set(rows))

        grid = {}
        for x in rows:
            for y in cols:
                grid[x, y] = "."

        for dict in dicts:
            row = dict["name"]
            for key, value in dict["data"].items():
                grid[row, key] = value

        max_model = 0
        for r in rows:
            if len(r) > max_model:
                max_model = len(r)
        max_model += 1

        max_col = 6
        for c in cols:
            if len(c) > max_col:
                max_col = len(c)
        max_col += 1

        with open(self.summary.get_path(), "wt") as f:
            row_string = ""
            row_string += "| {message: <{fill}}|".format(
                message="Models", fill=max_model
            )
            for c in cols:
                row_string += " {message: <{fill}}|".format(message=c, fill=max_col)
            f.write(row_string + "\n")
            for r in rows:
                row_string = ""
                row_string += "| {message: <{fill}}|".format(message=r, fill=max_model)
                for c in cols:
                    row_string += " {message: <{fill}}|".format(
                        message=grid[r, c], fill=max_col
                    )
                f.write(row_string + "\n")


class KaldiSummaryJob(PrintTableJob):
    def __init__(
        self,
        data,
        header,
        col_names=None,
        row_names=None,
        file_name="wer.txt",
        **kwargs,
    ):
        super().__init__(data, header, col_names, row_names, **kwargs)
        self.file_name = file_name

        self.summary = self.output_path("summary.txt")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        data = {}
        for report_dir, col, row in self.data:
            wer_file = os.path.join(tk.uncached_path(report_dir), self.file_name)
            data[(col, row)] = self.wer(wer_file)
        self.data = data
        super().run()

    @staticmethod
    def wer(path):
        with open(path, "rt") as f:
            for line in f:
                if line.startswith("%WER"):
                    wer = float(line.split()[1])
                    return "{}".format(wer)
        return None
