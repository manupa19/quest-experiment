import mlflow
from datetime import datetime

from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC


from experiments.dataloader import DataLoader
from experiments.config import ConfigHandler
from experiments.evaluation import evaluate_model
from experiments import Logger
from experiments.training import pipeline

from quest import QUESTInspired


logger_object = Logger()
logger = logger_object.get_logger()

config = ConfigHandler()

set_config(transform_output="pandas")


def main(experiment_name: str, run_name: str):

    mlflow.set_tracking_uri(f"sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config.get_data_config)
        mlflow.set_tag("tag", config.get_mlflow_tags)
        dataloader = DataLoader()
        data = dataloader.get_data()

        X_train, X_test, y_train, y_test = train_test_split(data.drop(config.get_target_column, axis=1)
                                                            , data[config.get_target_column], test_size=.2,
                                                            random_state=123, stratify=data[config.get_target_column])
        mlflow.sklearn.autolog()

        with mlflow.start_run(run_name='Classical without Feature Selection', nested=True):
            pipe = pipeline(data=X_train, classifier=RandomForestClassifier())
            pipe.fit(X_train, y_train)
            evaluate_model(pipe, X_test, y_test)

        with mlflow.start_run(run_name='Classical Feature Selection #1', nested=True):

            pipe = pipeline(data=X_train, classifier=RandomForestClassifier(),
                            step=('feature selection', SelectFromModel(LinearSVC(penalty="l2"))))
            pipe.fit(X_train, y_train)
            evaluate_model(pipe, X_test, y_test)

        with mlflow.start_run(run_name='Classical Feature Selection #2', nested=True):

            pipe = pipeline(data=X_train, classifier=RandomForestClassifier(),
                            step=('feature selection', SelectKBest(f_classif, k=10)))
            pipe.fit(X_train, y_train)
            evaluate_model(pipe, X_test, y_test)

        with mlflow.start_run(run_name='Classical Feature Selection #3', nested=True):

            pipe = pipeline(data=X_train, classifier=RFE(RandomForestClassifier(), verbose=True, step=5))
            pipe.fit(X_train, y_train)
            evaluate_model(pipe, X_test, y_test)

        with mlflow.start_run(run_name='Quest', nested=True):

            pipe = pipeline(data=X_train, classifier=RandomForestClassifier(),
                             step=('feature selection', QUESTInspired(**config.get_quest_params)))
            pipe.fit(X_train, y_train)
            evaluate_model(pipe, X_test, y_test)
            mlflow.log_params(config.get_quest_params)

        # mlflow.log_artifact(local_path=logger_object.log_filepath, artifact_path="log")


if __name__ == "__main__":
    experiment_name = "test"
    run_name = datetime.now().strftime("%Y%m%d+%H%M%S")

    main(experiment_name, run_name)





