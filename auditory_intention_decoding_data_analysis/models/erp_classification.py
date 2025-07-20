import pickle
from pathlib import Path
from typing import List

import mne
import numpy as np
import pandas as pd
import typer
from mne.decoding import CSP
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

from pyriemann.estimation import ERPCovariances, XdawnCovariances
from pyriemann.classification import MDM, FgMDM
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from auditory_intention_decoding_data_analysis.models.utils import file_to_label_file, read_labels_file, Prediction

mne.use_log_level("warning")
app = typer.Typer()


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

    # Riemannian pipeline

    # the ways of calculating riemann distance, riemann, logeuclid, euclid, kullback, wasserstein
    clf = make_pipeline(
        XdawnCovariances(classes=[0, 1], estimator='lwf'),
        #MDM(metric='riemann')
        # CSP(n_components=4, reg='ledoit_wolf'),
        TangentSpace(metric="riemann"),
        #StandardScaler(),
        SVC(kernel='poly')
        #LinearDiscriminantAnalysis()
    )

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    accuracies = []
    f1_scores = []
    predictions = []

    for i, (train_idx, test_idx) in enumerate(sss.split(data, labels)):
        X_train, y_train = data[train_idx], labels[train_idx]
        X_test, y_test = data[test_idx], labels[test_idx]
        taggers_test = taggers[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        accuracies.append(acc)
        f1_scores.append(f1)
        predictions.append(Prediction(predictions=y_pred, labels=y_test, taggers=taggers_test))

        print(f"[Split {i}] Accuracy: {acc:.4f}, F1: {f1:.4f}")

    output_folder.mkdir(exist_ok=True, parents=True)

    with open(output_folder / "accuracies.pkl", "wb") as f:
        pickle.dump(accuracies, f)

    with open(output_folder / "f1.pkl", "wb") as f:
        pickle.dump(f1_scores, f)

    with open(output_folder / "prediction.pkl", "wb") as f:
        pickle.dump(predictions, f)


if __name__ == '__main__':
    app()
