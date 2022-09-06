import logging
import argparse

_program = "tileai"
__version__ = "0.0.1"


logging.getLogger(__name__).addHandler(logging.NullHandler())


SubParsersAction = argparse._SubParsersAction