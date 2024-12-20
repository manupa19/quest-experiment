import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import ClassifierMixin, RegressorMixin


def pipeline(data: pd.DataFrame, classifier: ClassifierMixin or RegressorMixin,
             step: tuple = None) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), data.select_dtypes(include=['int', 'float']).columns),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), data.select_dtypes(include=['object']).columns)
        ],
        remainder='passthrough'
    )
    steps = [
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler())

    ]
    if step:
        steps.append(step)
    steps.append(('classifier', classifier))

    return Pipeline(steps=steps)
