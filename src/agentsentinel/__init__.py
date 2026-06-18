__version__ = "0.2.0"

import logging
from agentsentinel.utils.logger import setup_logger as setup_logger

logging.getLogger("agentsentinel").addHandler(logging.NullHandler())

