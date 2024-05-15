Please always report problems by filling an issue here on GitHub. This also covers reporting problems with the documentation (e.g. if sth is unclear).

**General rules:** 

    1. the code style suggestions of PyCharm should be followed

**Job general:** 

    1. If a class inherits directly/indirectly from the sisyphus class `Job` the class ends with "Job".
    2. In a Sisyphus job class the output variables start with `out_*`
    3. If a Job has non-trivial requirements it should have a `self.rqmt`

**Job constructor order:** 

    1. set inputs
    2. set outputs (should be prefixed with `out_`)
    3. define rqmt

**Job function order:** 

    1. `__init__`
    2. `tasks()`
    3. task functions
    4. helper functions
    5. class functions
    6. `__hash__`

Rasr-Jobs:
should always accept `crp`, `extra_config` + `extra_post_config`

Jobs that get a path to a specific software/toolkit (e.g. RETURNN or RASR) should have this
as last parameter in the `__init__()` definition.

**Import order:** 

    1. standard library
    2. external libraries (including sisyphus, yes I know it's not like that atm)
    3. recipies

**Some general things:**

    1. Don't break a jobs hash
    2. Really don't break a jobs hash
    3. very strong preference to avoid config files / scripts in the recipe repo itself. Everything should be code
    4. Jobs have to be deterministic. Especially the code in the Job itself and the computed output (files). If this is not possible at all, please add a clear warning.
    5. `sh` function should definitely not be used

[Black](https://github.com/psf/black) formatting:

The check for black formating is a fixed test case that will automatically run for new pull requests.
To prohibit errors before commiting,
it is recommended to add a git hook that will automatically perform the check before submission.

Create an executable `.git/hooks/pre-commit` with:

    #!/bin/bash
    set -eu
    black --check .
    exit 0

Corpus variable naming: in sisyphus setups there are different ways to hold the corpus information (str, Path, CorpusObject, RasrConfig). At the moment `corpus` covers all. To distinguish the different forms to hold the corpus information, we strongly encourage using the following variable naming:

    corpus_key -> str (basically corpus identifier)
    corpus_object -> CorpusObject
    corpus_file -> Path (would be filepath)
    corpus_config -> RasrConfig describing the corpus

    corpus any, but usage is discouraged
