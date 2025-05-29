import logging
logger = logging.getLogger(__name__)

def add(a, b):
    logger.info("JIT-compiled function called")
    return a + b
