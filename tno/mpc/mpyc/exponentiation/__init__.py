"""
Initialization of the exponentiation package.
"""

# Explicit re-export of all functionalities, such that they can be imported properly. Following
# https://www.python.org/dev/peps/pep-0484/#stub-files and
# https://mypy.readthedocs.io/en/stable/command_line.html#cmdoption-mypy-no-implicit-reexport
from .exponent import secure_pow as secure_pow
from .utils import convert_to_secfxp as convert_to_secfxp
from .utils import convert_to_secint as convert_to_secint

__version__ = "1.6.2"
