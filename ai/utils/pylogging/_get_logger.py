import logging


def get_logger(name: str):
    logging.basicConfig(level=logging.INFO)
    logger =  logging.getLogger(name)
    return logger
