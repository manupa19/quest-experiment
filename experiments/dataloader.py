import mlflow

import pandas as pd

from ydata_profiling import ProfileReport

from experiments.config import ConfigHandler
from experiments import Logger

logger = Logger().get_logger()


class DataLoader:

    def __init__(self):
        self.config_handler = ConfigHandler()
        self.data_path = self.config_handler.get_data_path

    def load_data(self):
        if self.data_path.endswith(".csv"):
            try:
                return pd.read_csv(self.data_path)
            except Exception:
                return pd.read_csv(self.data_path, delimiter=";", encoding="utf-8")
        elif self.data_path.endswith(".xlsx"):
            return pd.read_excel(self.data_path)

    def get_stats(self, data) -> None:
        logger.info(f"data shape {data.shape}")
        logger.info(f"target counts: {data[self.config_handler.get_target_column].value_counts()}")
        logger.info(f"Statistics of data at {self.data_path}: \n {data.info()}")
        mlflow.log_param("data shape", data.shape)
        mlflow.log_param("target counts", data[self.config_handler.get_target_column].value_counts())

    def get_eda(self, data) -> None:
        profile = ProfileReport(data, title=self.config_handler.get_data_file)
        profile.to_file(
            f"experiments/{self.config_handler.get_results_folder}/"
            f"{self.config_handler.get_data_file.split('.')[0]}_report.html")
        mlflow.log_artifact(
            f"experiments/{self.config_handler.get_results_folder}"
            f"/{self.config_handler.get_data_file.split('.')[0]}_report.html")

    def get_data(self) -> pd.DataFrame:
        data = self.load_data()
        self.get_stats(data)
        if bool(self.config_handler.get_eda_bool):
            self.get_eda(data)
        logger.info(f"Data loaded from {self.data_path}")
        return data


class DataSaver:

    def __init__(self, data, timestamp, tag=""):
        self.config_handler = ConfigHandler()
        self.data = data
        self.save_path = f"{self.config_handler.get_results_folder}/{timestamp}_{tag}_results"

    def save_data(self):
        self.data.to_csv(self.save_path+".csv", index=False, header=True)

    def save_data_as_excel(self):
        self.data.to_excel(self.save_path+".xlsx")

