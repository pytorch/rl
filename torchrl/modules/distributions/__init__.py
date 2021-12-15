from .continuous import *
from .discrete import *
from .utils import UNIFORM

from .continuous import __all__ as _all_continuous
from .discrete import __all__ as _all_discrete

distributions_maps = {
    distribution_class.lower(): eval(distribution_class)
    for distribution_class in _all_continuous + _all_discrete
}

