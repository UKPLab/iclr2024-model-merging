__all__ = ["CodeWrapper", "ReturnnConfig", "WriteReturnnConfigJob"]

import base64
import black
import inspect
import json
import os
import pickle
import pprint
import string
import sys
import textwrap

from sisyphus import *
from sisyphus.delayed_ops import DelayedBase
from sisyphus.hash import sis_hash_helper

from i6_core.util import instanciate_delayed

Path = setup_path(__package__)
Variable = tk.Variable


class CodeWrapper:
    """
    Can be used to insert direct "code" (not as quoted string) into the config dict
    """

    def __init__(self, code):
        self.code = code

    def __repr__(self):
        return self.code


class ReturnnConfig:
    """
    An object that manages a RETURNN config.

    It can be used to serialize python functions and class definitions directly from
    Sisyphus code and paste them into the RETURNN config file.

    """

    PYTHON_CODE = textwrap.dedent(
        """\
        #!rnn.py

        ${SUPPORT_CODE}

        ${PROLOG}
    
        ${REGULAR_CONFIG}
    
        locals().update(**config)
    
        ${EPILOG}
        """
    )

    GET_NETWORK_CODE = textwrap.dedent(
        """\
        import os
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        
        def get_network(epoch, **kwargs):
          from networks import networks_dict
          for epoch_ in sorted(networks_dict.keys(), reverse=True):
            if epoch_ <= epoch:
              return networks_dict[epoch_]
          assert False, \"Error, no networks found\"
        
        """
    )

    def __init__(
        self,
        config,
        post_config=None,
        staged_network_dict=None,
        *,
        python_prolog=None,
        python_prolog_hash=None,
        python_epilog="",
        python_epilog_hash=None,
        hash_full_python_code=False,
        sort_config=True,
        pprint_kwargs=None,
        black_formatting=True,
    ):
        """

        :param dict config: dictionary of the RETURNN config variables that are hashed
        :param dict post_config: dictionary of the RETURNN config variables that are not hashed
        :param None|dict[int, str|dict[str, Any]] staged_network_dict:
            dictionary of network dictionaries or
            any string that defines a network with `network = ...`
            (e.g. the return variable of get_ext_net_dict_py_code_str() from returnn_common),
            indexed by the desired starting epoch of the network stage.
            By enabling this, an additional "networks" folder will be created next to the config location.
        :param None|str|Callable|Class|tuple|list|dict python_prolog: str or structure containing str/callables/classes
            that should be pasted as code at the beginning of the config file
        :param None|Any python_prolog_hash: set a specific hash (str) or any type of hashable objects
            to overwrite the default hashing for python_prolog
        :param None|str|Callable|Class|tuple|list|dict python_epilog: str or structure containing
            str/callables/classes that should be pasted as code at the end of the config file
        :param None|Any python_epilog_hash: set a specific hash (str) or any type of hashable objects
            to overwrite the default hashing for python_epilog
        :param bool hash_full_python_code: By default, function bodies are not hashed. If set to True, the full content
            of python pro-/epilog is parsed and hashed.
        :param bool sort_config: If set to True, the dictionary part of the config is sorted by key
        :param dict|None pprint_kwargs: kwargs for pprint, e.g. {"sort_dicts": False} to print dicts in given order for
            python >= 3.8
        :param bool black_formatting: if true, the written config will be formatted with black
        """
        self.config = config
        self.post_config = post_config if post_config is not None else {}
        self.staged_network_dict = staged_network_dict
        self.python_prolog = python_prolog
        self.python_prolog_hash = python_prolog_hash
        if self.python_prolog_hash is None:
            if hash_full_python_code:
                self.python_prolog_hash = self.__parse_python(python_prolog)
            else:
                self.python_prolog_hash = python_prolog
        self.python_epilog = python_epilog
        self.python_epilog_hash = python_epilog_hash
        if self.python_epilog_hash is None:
            if hash_full_python_code:
                self.python_epilog_hash = self.__parse_python(python_epilog)
            else:
                self.python_epilog_hash = python_epilog
        self.hash_full_python_code = hash_full_python_code
        self.sort_config = sort_config
        self.pprint_kwargs = pprint_kwargs or {}
        if sys.version_info[:2] >= (3, 8):
            self.pprint_kwargs.setdefault("sort_dicts", sort_config)
        self.black_formatting = black_formatting

    def get(self, key, default=None):
        if key in self.post_config:
            return self.post_config[key]
        return self.config.get(key, default)

    def update(self, other):
        """
        updates a ReturnnConfig with an other ReturnnConfig:
          * config, post_config, and pprint_kwargs use dict.update
          * prolog, epilog, and hashes are concatenated
          * staged_network_dict, sort_config, and black_formatting are overwritten

        :param ReturnnConfig other:
        """
        self.config.update(other.config)
        self.post_config.update(other.post_config)
        if other.staged_network_dict is not None:
            self.staged_network_dict = other.staged_network_dict

        def join_code(my_code, other_code):
            if my_code is None or my_code == "":
                return other_code
            if other_code is None or other_code == "":
                return my_code
            if isinstance(my_code, str) and isinstance(other_code, str):
                return my_code + "\n" + other_code
            return [my_code, other_code]

        self.python_prolog = join_code(self.python_prolog, other.python_prolog)
        self.python_prolog_hash = join_code(
            self.python_prolog_hash, other.python_prolog_hash
        )
        self.python_epilog = join_code(self.python_epilog, other.python_epilog)
        self.python_epilog_hash = join_code(
            self.python_epilog_hash, other.python_epilog_hash
        )

        self.sort_config = other.sort_config
        self.pprint_kwargs.update(other.pprint_kwargs)
        self.black_formatting = other.black_formatting
        self.check_consistency()

    def _write_to_file(self, content, file_path):
        """
        write with optional black formatting

        :param str content:
        :param str config_path:
        """
        with open(file_path, "wt", encoding="utf-8") as f:
            if self.black_formatting:
                content = black.format_str(content, mode=black.Mode())
            f.write(content)

    def _write_network_stages(self, config_path):
        """
        write the networks of the staged network dict into a "networks" folder including
        the access dictionary in the init file

        :param str config_path:
        """
        config_dir = os.path.dirname(config_path)
        network_dir = os.path.join(config_dir, "networks")
        if not os.path.exists(network_dir):
            os.mkdir(network_dir)

        init_file = os.path.join(network_dir, "__init__.py")
        init_import_code = ""
        init_dict_code = "\n\nnetworks_dict = {\n"

        for epoch in self.staged_network_dict.keys():
            network_path = os.path.join(network_dir, "network_%i.py" % epoch)
            pp = pprint.PrettyPrinter(indent=2, width=150, **self.pprint_kwargs)
            network_definition = self.staged_network_dict[epoch]
            if isinstance(network_definition, dict):
                content = "\nnetwork = %s" % pp.pformat(network_definition)
            elif isinstance(network_definition, str):
                content = network_definition
            else:
                assert False, (
                    "Invalid entry type in staged_network_dict: %s, use str or dict"
                    % type(network_definition)
                )
            with open(network_path, "wt", encoding="utf-8") as f:
                if self.black_formatting:
                    content = black.format_str(content, mode=black.Mode())
                f.write(content)
            init_import_code += "from .network_%i import network as network_%i\n" % (
                epoch,
                epoch,
            )
            init_dict_code += "  %i: network_%i,\n" % (epoch, epoch)

        init_dict_code += "}\n"
        self._write_to_file(init_import_code + init_dict_code, init_file)

    def _serialize(self):
        """
        Serialize the main config
        :return:
        """
        self.check_consistency()
        config = self.config
        config.update(self.post_config)

        config = instanciate_delayed(config)

        config_lines = []
        unreadable_data = {}

        pp = pprint.PrettyPrinter(indent=2, width=150, **self.pprint_kwargs)
        for k, v in sorted(config.items()) if self.sort_config else config.items():
            if pprint.isreadable(v):
                config_lines.append("%s = %s" % (k, pp.pformat(v)))
            else:
                unreadable_data[k] = v

        if len(unreadable_data) > 0:
            config_lines.append("import json")
            json_data = json.dumps(unreadable_data).replace('"', '\\"')
            config_lines.append('config = json.loads("%s")' % json_data)
        else:
            config_lines.append("config = {}")

        python_prolog_code = self.__parse_python(self.python_prolog)
        python_epilog_code = self.__parse_python(self.python_epilog)

        support_code = ""
        if self.staged_network_dict:
            support_code += self.GET_NETWORK_CODE

        python_code = string.Template(self.PYTHON_CODE).substitute(
            {
                "SUPPORT_CODE": support_code,
                "PROLOG": python_prolog_code,
                "REGULAR_CONFIG": "\n".join(config_lines),
                "EPILOG": python_epilog_code,
            }
        )
        return python_code

    def write(self, path):
        """
        :param str path:
        """
        if self.staged_network_dict:
            self._write_network_stages(path)
        self._write_to_file(self._serialize(), path)

    def __parse_python(self, code, name=None):
        if code is None:
            return ""
        if isinstance(code, str):
            return code
        if isinstance(code, DelayedBase):
            assert self.hash_full_python_code is False, (
                "DelayedBase object can not be passed if `hash_full_python_code` is set, "
                "as this will cause breaking hashes"
            )
            return self.__parse_python(code.get())
        if isinstance(code, (tuple, list)):
            return "\n".join(self.__parse_python(c) for c in code)
        if isinstance(code, dict):
            return "\n".join(self.__parse_python(v, name=k) for k, v in code.items())
        if inspect.isfunction(code):
            try:
                return inspect.getsource(code)
            except OSError:
                # cannot get source, e.g. code is a lambda
                assert name is not None
                args = [
                    code.__code__.co_argcount,
                    code.__code__.co_kwonlyargcount,
                    code.__code__.co_nlocals,
                    code.__code__.co_stacksize,
                    code.__code__.co_flags,
                    code.__code__.co_code,
                    code.__code__.co_consts,
                    code.__code__.co_names,
                    code.__code__.co_varnames,
                    code.__code__.co_filename,
                    code.__code__.co_name,
                    code.__code__.co_firstlineno,
                    code.__code__.co_lnotab,
                    code.__code__.co_freevars,
                    code.__code__.co_cellvars,
                ]
                compiled = base64.b64encode(pickle.dumps(args)).decode("utf8")
                return (
                    "import types; import base64; import pickle; "
                    'code = types.CodeType(*pickle.loads(base64.b64decode("%s".encode("utf8")))); '
                    '%s = types.FunctionType(code, globals(), "%s")'
                    % (compiled, name, code.__name__)
                )
        if inspect.isclass(code):
            return inspect.getsource(code)
        raise RuntimeError("Could not serialize %s" % code)

    def check_consistency(self):
        """
        Check that there is no config key overwritten by post_config.
        Also check for parameters that should never be hashed.
        """
        for key in self.config:
            assert key not in self.post_config, (
                "%s in post_config would overwrite existing entry in config" % key
            )
        assert not (self.staged_network_dict and "network" in self.config)

        # list of parameters that should never be hashed
        disallowed_in_config = [
            "cleanup_old_models",
            "log_verbosity",
        ]
        for key in disallowed_in_config:
            assert self.config.get(key) is None, (
                "please define %s only as parameter in the post_config" % key
            )

    def _sis_hash(self):
        h = {
            "returnn_config": self.config,
            "python_epilog_hash": self.python_epilog_hash,
            "python_prolog_hash": self.python_prolog_hash,
        }
        if self.staged_network_dict:
            h["returnn_networks"] = self.staged_network_dict

        return sis_hash_helper(h)


class WriteReturnnConfigJob(Job):
    """
    Writes a ReturnnConfig into a .config file
    """

    def __init__(self, returnn_config):
        """

        :param ReturnnConfig returnn_config:
        """
        assert isinstance(returnn_config, ReturnnConfig)

        self.returnn_config = returnn_config

        self.out_returnn_config_file = self.output_path("returnn.config")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        self.returnn_config.write(self.out_returnn_config_file.get_path())
