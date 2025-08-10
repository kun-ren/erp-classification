import pickle
from pathlib import Path
from typing import List

import mne
import numpy as np
import pandas as pd
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

from pyriemann.estimation import ERPCovariances, XdawnCovariances
from pyriemann.classification import MDM, FgMDM, TSclassifier

from auditory_intention_decoding_data_analysis.models.utils import file_to_label_file, read_labels_file, Prediction

mne.use_log_level("warning")
app = typer.Typer()
import os
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

def shuffle_pipeline(pipeline, data, labels, taggers):
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    accuracies = []
    f1_scores = []
    predictions = []

    for i, (train_idx, test_idx) in enumerate(sss.split(data, labels)):
        X_train, y_train = data[train_idx], labels[train_idx]
        X_test, y_test = data[test_idx], labels[test_idx]
        taggers_test = taggers[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        accuracies.append(acc)
        f1_scores.append(f1)
        predictions.append(Prediction(predictions=y_pred, labels=y_test, taggers=taggers_test))

        print(f"[Split {i}] Accuracy: {acc:.4f}, F1: {f1:.4f}")

    overall_average = sum(accuracies) / len(accuracies)

    overall_f1 = sum(f1_scores) / len(f1_scores)

    print(f"[Overall Average] Accuracy: {overall_average:.4f}, F1: {overall_f1:.4f}")

    return accuracies, f1_scores, predictions


def write_training_results(output_folder, result, suffix=''):
    output_folder.mkdir(exist_ok=True, parents=True)

    with open(output_folder / ("accuracies-" + suffix + ".pkl"), "wb") as f:
        pickle.dump(result[0], f)

    with open(output_folder / ("f1-" + suffix + ".pkl"), "wb") as f:
        pickle.dump(result[1], f)

    with open(output_folder / ("prediction-" + suffix + ".pkl"), "wb") as f:
        pickle.dump(result[2], f)


@app.command()
def main(
        input_files: List[Path],
        output_folder: Path,
):
    print(f'Input file list:{input_files}')
    print(f'Output folder:{output_folder}')
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

    # 1. test the impact of filters.
    clf_riemann_space_with_filter = make_pipeline(
        XdawnCovariances(classes=[0, 1], estimator='lwf',),
        MDM(metric='riemann', n_jobs=4),
    )
    clf_riemann_space_without_filter = make_pipeline(
        ERPCovariances(classes=[0, 1], estimator='lwf', ),
        MDM(metric='riemann', n_jobs=4),
    )

    # 2. tagentSpace classification

    clf_tagent_space = make_pipeline(
        ERPCovariances(classes=[0, 1], estimator='lwf', ),
        TSclassifier(metric='riemann', tsupdate=True, clf=LogisticRegression(n_jobs=4),)
    )

    clf_reduce_dimension = make_pipeline(
        ERPCovariances(classes=[0, 1], estimator='lwf'),
        FgMDM(metric='riemann', tsupdate=True, n_jobs=4)   # FgMDM = FGDA + MDM
    )



    #clf_riemann_space_with_filter.fit(data)

    # todo change output folder
    # 1. test the impact of filters.
    result_filter = shuffle_pipeline(clf_riemann_space_with_filter, data, labels, taggers)
    write_training_results(output_folder, result_filter, 'xdawn_filter')

    result_without_filter = shuffle_pipeline(clf_riemann_space_without_filter, data, labels, taggers)
    write_training_results(output_folder, result_without_filter, 'no_filter')

    # 2. tagentSpace classification
    result_tagent_space = shuffle_pipeline(clf_tagent_space, data, labels, taggers)
    write_training_results(output_folder, result_tagent_space, 'tagentSpace')

    result_reduce_dimension = shuffle_pipeline(clf_reduce_dimension, data, labels, taggers)
    write_training_results(output_folder, result_reduce_dimension, 'FGDA')



if __name__ == '__main__':
    app()
