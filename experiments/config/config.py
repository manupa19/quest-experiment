import os
import yaml


class BasicConfigHandler:

    def __init__(self, path):
        self.path = path
        with open(self.path, encoding='utf-8') as config_file:
            self.config = yaml.safe_load(config_file)

    @property
    def get_config(self) -> dict:
        return self.config


class ConfigHandler(BasicConfigHandler):

    def __init__(self, path="experiments/config/config.yaml"):
        super().__init__(path=path)

    @property
    def get_data_config(self):
        return self.config['data']

    @property
    def get_data_folder(self):
        return self.get_data_config['folder']

    @property
    def get_data_file(self):
        return self.get_data_config['file']

    @property
    def get_data_path(self):
        return os.path.join(self.get_data_folder, self.get_data_file)

    @property
    def get_results_folder(self):
        return self.get_data_config['results']

    @property
    def get_target_column(self):
        return self.get_data_config['target_column']

    @property
    def get_other_config(self):
        return self.config['other']

    @property
    def get_mlflow_tags(self):
        return self.get_other_config['mlflow_tags']

    @property
    def get_eda_bool(self):
        return self.get_other_config['create_eda_report']

    @property
    def get_quest_config(self):
        return self.config['quest']

    @property
    def get_quest_params(self):
        return self.get_quest_config['params']


