"""
Automatic generation of the API files which include automodule definitions
"""

import os

exclude = {"autonet", "mod"}


def generate():
    if not os.path.exists("api"):
        os.mkdir("api")

    def makeapi(modname):
        """
        :param str modname:
        """
        fn = "api/%s.rst" % (modname[len("i6_core.") :] or "___base")
        if os.path.exists(fn):
            return
        f = open(fn, "w")
        title = ":mod:`%s`" % modname
        f.write("\n%s\n%s\n\n" % (title, "-" * len(title)))
        f.write(".. automodule:: %s\n\t:members:\n\t:undoc-members:\n\n" % modname)
        f.close()

    def scan_modules(modpath):
        """
        :param list[str] modpath:
        """
        # makeapi(".".join(modpath))
        path = "/".join(modpath)

        # First all sub packages.
        for fn in sorted(os.listdir(path)):
            print(fn)
            print(os.path.join(path, fn))
            if not os.path.isdir(os.path.join(path, fn)):
                continue
            if fn == "__pycache__":
                continue
            if (
                not os.path.exists(os.path.join(path, fn, "__init__.py"))
                and not fn == "i6_core"
            ):
                continue
            scan_modules(modpath + [fn])

        # Now all sub modules in this package.
        for fn in sorted(os.listdir(path)):
            if os.path.isdir(os.path.join(path, fn)):
                continue
            if not fn.endswith(".py"):
                continue
            if fn == "__init__.py":
                continue
            modname, _ = os.path.splitext(os.path.basename(fn))
            if modname in exclude:
                continue
            makeapi(".".join(modpath + [modname]))

    scan_modules(["i6_core"])
