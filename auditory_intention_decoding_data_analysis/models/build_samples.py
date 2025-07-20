import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import mne
import numpy as np
import typer
from loguru import logger

from auditory_intention_decoding_data_analysis.models.utils import file_to_label_file

mne.use_log_level("warning")
app = typer.Typer()


@dataclass
class Tagger:
    option: int
    target: int


#the tpye stimulus is collectively defined by your the modification method of audio and the type of the event

TRIGGERS = {
    "RAW":      Tagger(option=50, target=70),
    "AM":       Tagger(option=51, target=71),
    "FM":       Tagger(option=52, target=72),
    "BINAURAL": Tagger(option=56, target=76)
}


@app.command()
def main(
        input_file: Path,
        output_file: Path,
        tmin: float = 0,
        tmax: float = 3,
        sfreq: Optional[int] = None
):
    # tmin = float(tmin)
    # tmax = float(tmax)

    assert input_file.is_file()
    assert input_file.suffix == ".fif"
    assert output_file.name.endswith("-epo.fif")

    logger.info(f"Started building features (extracting mne Epochs from: {input_file}")
    logger.info(f"Parameters are: tmin={tmin}, tmax={tmax}, sfreq={sfreq}")
    raw: mne.io.Raw = mne.io.read_raw_fif(input_file)
    if sfreq is not None:
        raw.load_data()
        raw.resample(sfreq)

    logger.info("Extracting events")
    events, event_id = mne.events_from_annotations(raw)
    # mne Epochs cut raw eeg data into smaller pieces according to events,
    # with a span from event_time - tmin to event_time + tmax.
    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, baseline=(None, 0), event_repeated="drop")

    epochs_options: Dict[str, mne.Epochs] = {trigger: epochs[f"Stimulus/S {TRIGGERS[trigger].option}"] for trigger in
                                             TRIGGERS}
    epochs_targets: Dict[str, mne.Epochs] = {trigger: epochs[f"Stimulus/S {TRIGGERS[trigger].target}"] for trigger in
                                             TRIGGERS}

    epochs_combined = mne.epochs.concatenate_epochs([*epochs_options.values(), *epochs_targets.values()])
    labels = np.array([0] * sum([len(eo) for eo in epochs_options.values()]) +
                      [1] * sum([len(et) for et in epochs_targets.values()]))
    taggers = (sum([[tagger] * len(epochs_options[tagger]) for tagger in epochs_options.keys()], []) +
               sum([[tagger] * len(epochs_targets[tagger]) for tagger in epochs_targets.keys()], []))

    logger.info("Saving epochs")
    epochs_combined.save(output_file, overwrite=True)
    with open(file_to_label_file(output_file), "w") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["Target", "Tagger"])
        writer.writerows([[int(label), tagger] for label, tagger in zip(labels, taggers)])


if __name__ == '__main__':
    app()
