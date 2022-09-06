import logging

from tileai.api import train

# define the version before the other imports since these need it


logging.getLogger(__name__).addHandler(logging.NullHandler())