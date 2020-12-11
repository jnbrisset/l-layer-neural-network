import logging
from time import time

LOG_FORMAT = '%(levelname)s %(asctime)s Line: %(lineno)d - %(message)s'
LOG_FILENAME = 'temp/{}.log'.format(int(time()))
# noinspection PyArgumentList
logging.basicConfig(
    level='DEBUG',
    format=LOG_FORMAT,
    handlers=[logging.FileHandler(LOG_FILENAME),
              logging.StreamHandler()
              ]
)
logger = logging.getLogger('main_dev')
