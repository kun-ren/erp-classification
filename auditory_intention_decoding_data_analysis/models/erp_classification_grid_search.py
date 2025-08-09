import inspect
import pickle
from pathlib import Path
from typing import List

import mne
import numpy as np
import pandas as pd
import typer
from mne.decoding import CSP
from pyriemann.spatialfilters import Xdawn, SPoC, BilinearFilter
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score, f1_score

from pyriemann.estimation import ERPCovariances, XdawnCovariances, Covariances
from pyriemann.classification import MDM, FgMDM, TSclassifier, SVC
from sklearn.preprocessing import StandardScaler

from auditory_intention_decoding_data_analysis.models.utils import file_to_label_file, read_labels_file, Prediction

mne.use_log_level("warning")
app = typer.Typer()


def make_filter(filter_cls, nfilter=None):
    if nfilter is not None:
        return filter_cls(nfilter=nfilter)
    else:
        return filter_cls()


class SpatialFilterWrapper(BaseEstimator):
    def __init__(self, metric='riemann', filter_cls=Xdawn, nfilter=2, classes=None, estimator='lwf'):
        self.filter_cls = filter_cls
        self.nfilter = nfilter
        self.classes = classes
        self.metric = metric
        self.estimator = estimator
        self.model = None

    def fit(self, X, y=None):
        sig = inspect.signature(self.filter_cls)
        valid_args = {
        }
        if 'metric' in sig.parameters:
            valid_args['metric'] = self.metric
        if 'classes' in sig.parameters:
            valid_args['classes'] = self.classes
        if 'estimator' in sig.parameters:
            valid_args['estimator'] = self.estimator
        if 'nfilter' in sig.parameters:
            valid_args['nfilter'] = self.nfilter

        self.model = self.filter_cls(**valid_args)
        self.model.fit(X, y)
        return self

    def transform(self, X):
        return self.model.transform(X)


class CovarianceWrapper(BaseEstimator):
    def __init__(self, cov_cls=Covariances, estimator='lwf', classes=None):
        self.cov_cls = cov_cls
        self.estimator = estimator
        self.model = None
        self.classes = classes

    def fit(self, X, y=None):
        sig = inspect.signature(self.cov_cls)
        valid_args = {
        }

        if 'classes' in sig.parameters:
            valid_args['classes'] = self.classes
        if 'estimator' in sig.parameters:
            valid_args['estimator'] = self.estimator

        self.model = self.cov_cls(**valid_args)
        self.model.fit(X, y)
        return self

    def transform(self, X):
        return self.model.transform(X)


class ClassifierWrapper(BaseEstimator):
    def __init__(self, metric='riemann', ts_clf=LinearDiscriminantAnalysis(), svm_kernel='riemann', svm_c=1,
                 clf_cls=MDM, sample_weight=None):
        self.clf_cls = clf_cls
        self.model = None
        self.metric = metric
        self.ts_clf = ts_clf
        self.svm_kernel = svm_kernel
        self.svm_c = svm_c

    def fit(self, X, y):
        sig = inspect.signature(self.clf_cls)
        valid_args = {
        }
        if 'metric' in sig.parameters:
            valid_args['metric'] = self.metric
        if 'clf' in sig.parameters:
            valid_args['clf'] = self.ts_clf
        if 'kernel_fct' in sig.parameters:
            valid_args['kernel_fct'] = self.svm_kernel
        if 'C' in sig.parameters:
            valid_args['C'] = self.svm_c

        self.model = self.clf_cls(**valid_args)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


@app.command()
def main(
        input_files: List[Path],
        output_folder: Path,
):
    print(f'Input file list:{input_files}')
    labels_files = [file_to_label_file(input_file) for input_file in input_files]

    assert all(input_file.name.endswith("-epo.fif") for input_file in input_files)
    assert all(labels_file.exists() for labels_file in labels_files)

    # load training data and labels
    epochs = [mne.read_epochs(input_file, preload=True) for input_file in input_files]
    assert all(epoch.info["sfreq"] == 128 for epoch in epochs)

    epochs_combined: mne.Epochs = mne.concatenate_epochs(epochs)
    labels_combined_df = pd.concat([read_labels_file(file) for file in labels_files])
    labels = labels_combined_df["Target"].to_numpy()
    taggers = np.array(labels_combined_df["Tagger"])

    data = epochs_combined.get_data()  # shape = (n_trials, n_channels, n_times)

    # param_grid = {
    #     # 'spatialfilter__metric': ['riemann', 'logeuclid', 'kullback_sym', 'wasserstein'],
    #     # 'spatialfilter__filter_cls': [Xdawn, CSP, SPoC, BilinearFilter],
    #     # 'spatialfilter__nfilter': [1, 2, 3, 4],
    #     # 'spatialfilter__estimator': ['oas', 'lwf', 'scm'],
    #     'cov__cov_cls': [Covariances, ERPCovariances],
    #     'cov__estimator': ['oas', 'lwf', 'scm'],
    #     'clf__clf_cls': [MDM, FgMDM, TSclassifier, SVC],
    #     'clf__metric': ['riemann', 'logeuclid', 'kullback_sym', 'wasserstein'],
    #     'clf__ts_clf': [LinearDiscriminantAnalysis(), LogisticRegression(), RidgeClassifier()],
    #     'clf__svm_kernel': ['riemann', 'logeuclid'],
    #     'clf__svm_c': [0.01, 0.1, 1, 10, 100]
    # }

    param_grid_riemann_space = {
        'cov__cov_classes': [Covariances, ERPCovariances],
        'cov__estimator': ['lwf'],
        'clf__classifier': [MDM, FgMDM],
        'clf__metric': ['riemann', ],
    }

    param_grid_tagent_space = {
        'cov__cov_cls': [Covariances, ERPCovariances],
        'cov__estimator': ['lwf'],
        'clf__clf_cls': [TSclassifier],
        'clf__metric': ['riemann', ],
        'clf__ts_clf': [LinearDiscriminantAnalysis(), LogisticRegression()],
        'clf__svm_kernel': ['riemann', ],
        'clf__svm_c': [0.01, 0.1, ]
    }

    pipe = Pipeline([
        # ('spatialfilter', SpatialFilterWrapper(classes=[0, 1], estimator='lwf')),
        ('cov', CovarianceWrapper()),
        ('clf', ClassifierWrapper())
    ])

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    gs_riemann_space = GridSearchCV(pipe, param_grid, cv=sss, scoring='accuracy', n_jobs=2)
    # gs_tagent_space = GridSearchCV(estimator=)

    gs_riemann_space.fit(data, labels)

    print("Best accuracy:", gs_riemann_space.best_score_)
    print("Best parameters:")
    for k, v in gs_riemann_space.best_params_.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    app()
