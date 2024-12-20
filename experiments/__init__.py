
import os
import sys
import logging
from datetime import datetime


class Logger:

    def __init__(self):
        self.logging_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.log_dir = "logs"
        self.folder = datetime.today().strftime("%Y-%m-%d")
        self.file_tag = datetime.now().strftime("%Y%m%d+%H%M%S")
        self.log_folder = os.path.join(self.log_dir, self.folder)
        self.log_filepath = os.path.join(self.log_folder, f"log_{self.file_tag}.log")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.log_folder, exist_ok=True)

    def _configure_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format=self.logging_str,
            handlers=[
                logging.FileHandler(self.log_filepath),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging

    def get_logger(self):
        logging = self._configure_logger()
        return logging.getLogger(__name__)

