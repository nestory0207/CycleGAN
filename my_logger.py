import os
import logging
from datetime import datetime

excute_file_path = os.path.abspath(__file__)
file_name = os.path.basename(excute_file_path)
dir_name = os.path.dirname(file_name)

def make_dir(dir_path):    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        
class MyLogger():
    logger_name = "UNIT"
    def __init__(self, log_dir_path, log_fname="SynDiff.log", logger_name=logger_name):
        self.log_dir_path = os.path.join(log_dir_path)
        if os.path.exists(self.log_dir_path):
            make_dir(self.log_dir_path)
        self.log_path = os.path.join(self.log_dir_path, log_fname)
        self.logger = self.create_logger(logger_name)

    def create_logger(self, logger_name=logger_name):
        # create logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        if len(logger.handlers) > 0:
            return logger # Logger already exists

        formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s")

        # create handlers
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger           
