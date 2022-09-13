import logging
import argparse
import sys

_program = "tileai"
__version__ = "0.0.2"



rootLogger = logging.getLogger(__name__)

#logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
#fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
#fileHandler.setFormatter(logFormatter)
#rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
rootLogger.addHandler(consoleHandler)

#logging.getLogger(__name__).addHandler(logging.NullHandler())


SubParsersAction = argparse._SubParsersAction