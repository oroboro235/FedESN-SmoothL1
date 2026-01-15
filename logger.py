# This file is used to log the training process and the testing results.

import time
import logging
import config 


class myLogger:
    def __init__(self, name: str, filename=None):
        self.logger = logging.getLogger(name)

        self.logger.setLevel(logging.INFO)
        # if the filename is none, then named with time
        log_path = config.path.log_path + (filename if filename else time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + ".log")
        handler = logging.FileHandler(log_path)

        formatter = logging.Formatter("%(message)s \n")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' + name)
        

    def info(self, message):
        self.logger.info(message)

if __name__ == "__main__":
    # test logger
    logger = myLogger("test2", "test.log")
    logger.info("test message1")
    logger.info("test message2")
    logger.info("test message3")

