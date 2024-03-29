import datetime
import logging
import os

PathLike = str | bytes | os.PathLike

def get_logger(modname: str, loglevel=logging.DEBUG) -> logging.Logger:
    logger  = logging.getLogger(modname)
    handler = logging.StreamHandler()
    handler.setLevel(loglevel)
    logger.setLevel(loglevel)
    logger.addHandler(handler)
    logger.propagate = False

    file_handler = logging.FileHandler(filename="vtuber-scraper.log", encoding="utf-8")
    file_handler.setLevel(loglevel)
    logger.addHandler(file_handler)

    return logger

def timestamp_to_str(timestamp: int) -> str:
    timestamp: datetime.datetime = datetime.datetime.fromtimestamp(timestamp / 1000)
    return timestamp.isoformat()
