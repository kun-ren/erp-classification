import pickle
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from auditory_intention_decoding_data_analysis.external.EEGModels import EEGNet
from auditory_intention_decoding_data_analysis.models.utils import Prediction, file_to_label_file, read_labels_file

mne.use_log_level("warning")
app = typer.Typer()


def eegnet_reshape(data: np.ndarray) -> np.ndarray:
    return data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))


@app.command()
def main(
        input_files: List[Path],
        output_folder: Path,
        n_epochs: int
):
    print(f'Input file list:{input_files}')
    labels_files = [file_to_label_file(input_file) for input_file in input_files]

    assert all(input_file.name.endswith("-epo.fif") for input_file in input_files)
    assert all(labels_file.exists() for labels_file in labels_files)

    epochs = [mne.read_epochs(input_file) for input_file in input_files]
    assert all(epoch.info["sfreq"] == 128 for epoch in epochs)

    epochs_combined: mne.Epochs = mne.concatenate_epochs(epochs)
    labels_combined_df = pd.concat([read_labels_file(file) for file in labels_files])
    labels_onehot = np.stack([-labels_combined_df["Target"] + 1, labels_combined_df["Target"]], axis=1)
    taggers = np.array(labels_combined_df["Tagger"])

    data = epochs_combined.get_data()

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    sss.get_n_splits(data, labels_onehot)

    splits = sss.split(data, labels_onehot)

    histories = []
    accuracies = []
    f1_scores = []
    predictions = []
    for i, split in enumerate(splits):
        data_train = data[split[0]]
        labels_train = labels_onehot[split[0]]

        data_test = data[split[1]]
        labels_test = labels_onehot[split[1]]
        taggers_test = taggers[split[1]]

        scaler = mne.decoding.Scaler(scalings="median")
        data_train_scaled = scaler.fit_transform(data_train)
        data_test_scaled = scaler.transform(data_test)

        data_train_scaled_shaped = eegnet_reshape(data_train_scaled)
        data_test_scaled_shaped = eegnet_reshape(data_test_scaled)

        model: Model = EEGNet(2, 126, int(3.5 * 128 + 1))  # TODO: parameters
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        history = model.fit(data_train_scaled_shaped, labels_train, epochs=n_epochs, batch_size=32, verbose=1)
        histories.append(history.history)

        argmax_labels = np.argmax(labels_test, axis=1)
        argmax_predictions = np.argmax(model.predict(data_test_scaled_shaped), axis=1)
        accuracies.append(metrics.accuracy_score(argmax_labels, argmax_predictions))
        f1_scores.append(metrics.f1_score(argmax_labels, argmax_predictions, average='weighted'))

        prediction = Prediction(predictions=argmax_predictions, labels=argmax_labels, taggers=taggers_test)
        predictions.append(prediction)

        model.save(output_folder / f"model-split-{i}.keras")

    with open(output_folder / "history.pkl", "wb") as file:
        pickle.dump(histories, file)

    with open(output_folder / "accuracies.pkl", "wb") as file:
        pickle.dump(accuracies, file)

    with open(output_folder / "f1.pkl", "wb") as file:
        pickle.dump(f1_scores, file)

    with open(output_folder / "prediction.pkl", "wb") as file:
        pickle.dump(predictions, file)

    history_accuracies = pd.DataFrame([history["accuracy"] for history in histories]).melt()
    history_loss = pd.DataFrame([history["loss"] for history in histories]).melt()

    plt.figure(figsize=(20, 10))
    sns.lineplot(history_accuracies, x="variable", y="value", label="accuracy")
    sns.lineplot(history_loss, x="variable", y="value", label="loss")
    plt.legend()
    plt.savefig(output_folder / "training_plot.png")
    plt.close()

    pass


if __name__ == '__main__':
    app()
