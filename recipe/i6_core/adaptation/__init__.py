from .linear_adaptation_layer import *
from .ubm import *

import importlib

bob_spec = importlib.util.find_spec("bob")
if bob_spec is not None:
    from .ivector import *
else:
    import logging

    logging.debug(
        'Warning (adaptation.__init__.py): "bob" was not found. I-vector training and extraction will not be available.'
    )
