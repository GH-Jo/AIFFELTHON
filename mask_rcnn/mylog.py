import logging
import os

_logger=None # logger
def make_logger(exp_dir_path:str):
    global _logger
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    fileHandler = logging.handlers.RotatingFileHandler(filename=os.path.join(exp_dir_path,'exp.log'),maxBytes=100*1024*1024,backupCount=10)
    formatter = logging.Formatter('[%(asctime)s %(filename)s:%(lineno)s] %(message)s')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    #logger.addHandler(TqdmToLoggerHandler())
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level = logging.DEBUG)
    logger.debug('exp.log start')
    _logger = logger
    return logger

def print_log(msg):
    global _logger
    if _logger is not None:
        _logger.debug(msg)