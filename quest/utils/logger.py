import logging.handlers
import logging


def logger(name: str = __name__):
    _logger = logging.getLogger(name)
    level = logging.DEBUG
    _logger.setLevel(level)
    sh = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    sh.setFormatter(formatter)
    _logger.addHandler(sh)
    return _logger

