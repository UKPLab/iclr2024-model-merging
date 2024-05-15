__all__ = [
    "RasrConfig",
    "build_config_from_mapping",
    "ConfigBuilder",
    "StringWrapper",
    "WriteRasrConfigJob",
]

import itertools as it

from sisyphus import Job, Task

from i6_core import util
from i6_core.rasr.command import RasrCommand


class RasrConfig:
    """
    Used to store a Rasr configuration
    """

    def __init__(self, prolog="", prolog_hash="", epilog="", epilog_hash=""):
        """
        :param string prolog: A string that should be pasted as code at the
            beginning of the config file
        :param string epilog: A string that should be pasted as code at the end of
            the config file
        :param string prolog_hash: sets a specific hash for the prolog
        :param string epilog_hash: sets a specific hash for the epilog
        """

        self.__dict = {}
        # special value that is used if there is a subtree where the root has also a value
        # e.g. recognizer.lm-lookahead = True
        #      recognizer.lm-lookahead.history-limit = 1
        self._value = None

        self._prolog = prolog
        self._prolog_hash = prolog_hash if prolog_hash else prolog
        self._epilog = epilog
        self._epilog_hash = epilog_hash if epilog_hash else epilog

    # TODO: investigate if normal (deep)copy can be used here
    def _copy(self):
        result = RasrConfig()
        result._value = self._value
        for k, v in self.__dict.items():
            if isinstance(v, RasrConfig):
                result.__dict[k] = v._copy()
            else:
                result.__dict[k] = v
        return result

    def _update(self, x):
        """
        :param RasrConfig|None x:
        """
        if x is None:
            return
        assert isinstance(x, RasrConfig)
        d = self.__dict
        if x._value is not None:
            self._value = x._value
        for k, v in x.__dict.items():
            if k not in d:
                d[k] = v._copy() if isinstance(v, RasrConfig) else v
            else:
                if isinstance(d[k], RasrConfig):
                    if isinstance(v, RasrConfig):
                        d[k]._update(v)
                    else:
                        d[k]._value = v
                else:
                    if isinstance(v, RasrConfig):
                        _val = v._value if v._value is not None else d[k]
                        d[k] = v._copy()
                        d[k]._value = _val
                    else:
                        d[k] = v

    def _items(self):
        return self.__dict.items()

    def _get(self, name, default=None):
        try:
            return self.__dict[name] if len(name) > 0 else self._value
        except KeyError:
            return default

    def _getter(self, name):
        if type(name) == str:
            name = name.split(".")
        if len(name) == 1:
            return self[name[0]]
        return self[name[0]]._getter(name[1:])

    def _set(self, name, value):
        if type(name) == str:
            name = name.split(".")
        if len(name) == 1:
            self[name[0]] = value
        else:
            self[name[0]]._set(name[1:], value)

    def __getitem__(self, name):
        if type(name) == str and "." in name:
            return self._getter(name)
        if len(name) == 0:
            return self._value
        d = self.__dict
        if name not in d:
            d[name] = RasrConfig()
        return d[name]

    def __setitem__(self, name, value):
        if "." in name:
            self._set(name, value)
        else:
            if isinstance(value, RasrConfig):
                if name in self.__dict:
                    if isinstance(self.__dict[name], RasrConfig):
                        self.__dict[name]._update(value)
                    else:
                        _val = (
                            value._value
                            if value._value is not None
                            else self.__dict[name]
                        )
                        self.__dict[name] = value._copy()
                        self.__dict[name]._value = _val
                else:
                    self.__dict[name] = value._copy()
            else:
                if name in self.__dict and isinstance(self.__dict[name], RasrConfig):
                    self.__dict[name]._value = value
                else:
                    self.__dict[name] = value

    def __delitem__(self, name):
        if name in self.__dict:
            del self.__dict[name]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        name = name.replace("_", "-")
        return self.__getitem__(name)

    def __setattr__(self, name, value):
        if name.startswith("_"):  # ignore everything starting with an underscore
            super().__setattr__(name, value)
        else:
            name = name.replace("_", "-")
            self[name] = value

    def __delattr__(self, name):
        if name.startswith("_"):
            return
        name = name.replace("_", "-")
        if name in self.__dict:
            del self.__dict[name]

    def __dir__(self):
        return self.__dict.keys()

    def __iter__(self):
        return self.__dict.__iter__()

    def __str__(self, parent_prefix="--"):
        l = []
        if self._value is not None:
            l.append("%s=%s" % (parent_prefix[:-1], self.__print_value(self._value)))
        for k, v in self.__dict.items():
            if isinstance(v, RasrConfig):
                l.append(v.__str__(parent_prefix + k + "."))
            elif v is None:
                pass
            else:
                l.append("%s%s=%s" % (parent_prefix, k, self.__print_value(v)))
        return " ".join(l)

    def html(self):
        return repr(self).replace("\n", "<br/>")

    def __repr_helper__(self):
        result = []
        for k, v in self.__dict.items():
            if isinstance(v, RasrConfig):
                if v._value is not None:
                    result.append(("", k, self.__print_value(v._value)))
                children = v.__repr_helper__()
                if len(children) <= 1:
                    for c in children:
                        result.append(
                            (
                                "",
                                ("%s.%s.%s" % (k, c[0], c[1])).replace("..", "."),
                                c[2],
                            )
                        )
                else:
                    for c in children:
                        category = "%s.%s" % (k, c[0]) if len(c[0]) > 0 else k
                        result.append((category, c[1], c[2]))
            elif v is not None:
                result.append(("", k, self.__print_value(v)))
        return result

    def __repr__(self):
        buf = []
        if self._prolog:
            assert isinstance(self._prolog, str), "prolog must be in a string format"
            buf.append(self._prolog)
        l = self.__repr_helper__()
        l.sort()

        for k, g in it.groupby(l, lambda t: t[0]):
            g = list(g)
            if len(k) > 0:
                if len(buf) > 0:
                    buf.append("")
                buf.append("[%s]" % k)

            max_len = max(map(lambda t: len(t[1]), g))
            for t in g:
                buf.append("%-*s = %s" % (max_len, t[1], t[2]))
        if self._epilog:
            assert isinstance(self._epilog, str), "epilog must be in a string format"
            buf.append(self._epilog)
        return "\n".join(buf)

    def __sis_state__(self):
        result = {"tree": self.__dict, "value": self._value}
        if self._prolog_hash:
            result["prolog_hash"] = self._prolog_hash
        if self._epilog_hash:
            result["epilog_hash"] = self._epilog_hash
        return result

    @staticmethod
    def __print_value(val):
        val = util.get_val(val)
        if type(val) == bool:
            return "yes" if val else "no"
        if type(val) == list:
            return " ".join(RasrConfig.__print_value(e) for e in val)
        return str(val)


def build_config_from_mapping(crp, mapping, include_log_config=True, parallelize=False):
    """
    :param rasr.crp.CommonRasrParameters crp:
    :param dict[str,str|list[str]] mapping:
    :param bool include_log_config:
    :param bool parallelize:
    :return: config, post_config
    :rtype: (RasrConfig, RasrConfig)
    """
    config = RasrConfig()
    post_config = RasrConfig()

    if include_log_config:
        config._update(crp.log_config)
        post_config._update(crp.log_post_config)

    for mkey in ["corpus", "lexicon", "acoustic_model", "language_model", "recognizer"]:
        if mkey not in mapping:
            continue
        keys = mapping[mkey]
        if type(keys) == str:
            keys = (keys,)
        for key in keys:
            c = getattr(crp, "%s_config" % mkey)
            if c is not None:
                config[key] = c

            c = getattr(crp, "%s_post_config" % mkey)
            if c is not None:
                post_config[key] = c

            if mkey == "corpus" and parallelize:
                if crp.segment_path is not None:
                    config[key].segments.file = crp.segment_path
                elif crp.concurrent > 1:
                    config[key].partition = crp.concurrent
                    config[key].select_partition = "$(TASK)"

    if crp.python_home is not None:
        post_config["*"].python_home = crp.python_home
    if crp.python_program_name is not None:
        post_config["*"].python_program_name = crp.python_program_name

    return config, post_config


class ConfigBuilder:
    def __init__(self, defaults):
        self.defaults = defaults

    def __call__(self, **kwargs):
        result = RasrConfig()

        for k, v in self.defaults.items():
            result[k] = v

        for k, v in kwargs.items():
            nk = k.replace("_", "-")
            result[nk] = v

        return result


class StringWrapper:
    """
    Deprecated, please use e.g. DelayedFormat directly from Sisyphus
    Example for wrapping commands:

        command = DelayedFormat("{} -a -l en -no-escape", tokenizer_binary)

    Example for wrapping/combining paths:

        pymod_config = DelayedFormat("epoch:{},action:forward,configfile:{}", model.epoch, model.returnn_config_file)

    Example for wrapping even function calls:

        def cut_ending(path):
            return path[: -len(".meta")]

        def foo():
            [...]
            config.loader.saved_model_file = DelayedFunction(returnn_model.model, cut_ending)
    """

    def __init__(self, string, hidden=None):
        """

        :param str string: some string based on the hashing object
        :param Any hidden: hashing object
        """
        self.string = string
        self.hidden = hidden

    def __str__(self):
        return self.string


class WriteRasrConfigJob(RasrCommand, Job):
    """
    Write a RasrConfig object into a .config file
    """

    def __init__(self, config, post_config):
        """
        :param RasrConfig config: RASR config part that is hashed
        :param RasrConfig post_config: RASR config part that is not hashed
        """
        self.config = config
        self.post_config = post_config

        self.out_config = self.output_path("rasr.config")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        self.write_config(
            config=self.config,
            post_config=self.post_config,
            filename=self.out_config.get_path(),
        )

    @classmethod
    def hash(cls, kwargs):
        d = {"config": kwargs["config"]}
        return super().hash(d)
