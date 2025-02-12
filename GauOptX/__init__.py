import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging

logger = logging.getLogger(__name__)

from GauOptX.core.task.space import Design_space
from . import core
from . import methods
from . import util
from . import interface
from . import models
from . import acquisitions
from . import optimization
try:
    from . import objective_examples
    from . import objective_examples as fmodels
except ImportError as e:
    logger.warning("Could not import examples: {}".format(e))

from .__version__ import __version__
