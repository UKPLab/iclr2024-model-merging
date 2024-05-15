**i6_core** is a "recipe" collection for the [Sisyphus](https://github.com/rwth-i6/sisyphus) workflow manager
containing jobs to build speech recognition, machine translation or text-to-speech pipelines using
 the toolkits [RETURNN](https://github.com/rwth-i6/returnn) and [RASR](https://github.com/rwth-i6/rasr).

Status: stable

Please note that some exotic Jobs, especially those under `adaptation` are not fully tested.

# Installation

As the purpose of `i6_core` is to be used as recipe for
[Sisyphus](https://github.com/rwth-i6/sisyphus),
it is sufficient to clone it into the `recipe/` sub-folder of a Sisyphus setup,
e.g. in the setup root call:

`git clone https://github.com/rwth-i6/i6_core recipe/i6_core`


# Contributing

Before contributing to `i6_core`, please have a close look at
[CONTRIBUTING.md](https://github.com/rwth-i6/i6_core/blob/main/CONTRIBUTING.md)

Code style: https://github.com/psf/black

Please coordinate large changes with the maintainers, which are:
 - Nick Rossenbach [@JackTemaki](https://github.com/JackTemaki)
 - Wilfried Michel [@michelwi](https://github.com/michelwi)
 - Eugen Beck [@curufinwe](https://github.com/curufinwe)

# Testing

See [tests/README.md](https://github.com/rwth-i6/i6_core/blob/main/tests/README.md)

# License

All Source Code in this Project is subject to the terms of the Mozilla
Public License, v. 2.0. If a copy of the MPL was not distributed with
this file, You can obtain one at http://mozilla.org/MPL/2.0/.
