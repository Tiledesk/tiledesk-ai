import logging
import sys



from tileai.api import train, query
from tileai.httpapi import run


# define the version before the other imports since these need it

rootLogger = logging.getLogger(__name__)

#logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
#fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
#fileHandler.setFormatter(logFormatter)
#rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
rootLogger.addHandler(consoleHandler)

#logging.getLogger(__name__).addHandler(logging.NullHandler())
