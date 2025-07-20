import csv
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mne
import numpy as np
import typer
from loguru import logger

from auditory_intention_decoding_data_analysis.model import ETagger, TRIGGER_OFFSETS, Triggers

mne.use_log_level("warning")
app = typer.Typer()


def get_trigger_from_annotation(annotation: Dict[str, Any]) -> Optional[Tuple[int, float]]:
    stimulus = annotation["description"]
    match = re.search(r"Stimulus/S( )*", stimulus)

    if match is None:
        return None

    trigger_num = stimulus[match.end():] # extract the number behind Stimulus/S

    try:
        return int(trigger_num), float(annotation['onset'])
    except ValueError:
        return None


def process_michal_ses_01(input_file: Path, raw: mne.io.Raw):
    # This first recording needs further processing, as the triggers changed since then
    with open(input_file.parent / "taggers.csv", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)  # skip header

        tagger_sequence = list(map(lambda ele: ETagger.get_tagger(ele[0]), reader))

    counter = 0
    new_annotations = mne.Annotations(np.array([]), np.array([]), np.array([]))
    for annotation in raw.annotations:
        trigger = get_trigger_from_annotation(annotation)
        if trigger is None:
            continue

        if trigger[0] == Triggers.OPTION_START_LEGACY:
            offset = TRIGGER_OFFSETS.index(tagger_sequence[counter // 3])
            annotation["description"] = f"Stimulus/S {Triggers.OPTION_START_OFFSET + offset}"
            counter += 1
        elif trigger[0] == Triggers.TARGET_START_LEGACY:
            offset = TRIGGER_OFFSETS.index(tagger_sequence[counter // 3])
            annotation["description"] = f"Stimulus/S {Triggers.TARGET_START_OFFSET + offset}"
            counter += 1
        new_annotations.append(annotation['onset'], annotation['duration'], annotation['description'])

    new_annotations.rename({
        f"Stimulus/S{Triggers.OPTION_END_LEGACY}": f"Stimulus/S {Triggers.OPTION_END}",
        f"Stimulus/S{Triggers.TARGET_END_LEGACY}": f"Stimulus/S {Triggers.TARGET_END}"})

    raw.set_annotations(new_annotations)


@app.command()
def main(
        input_file: Path,
        output_file: Path,
        re_reference: bool
):
    assert input_file.is_file()
    assert input_file.suffix == ".vhdr"
    assert output_file.suffix == ".fif"

    raw = mne.io.read_raw_brainvision(input_file)

    logger.info("Loading channels and setting montage.")
    raw.rename_channels({'O9': 'I1', 'O10': 'I2'})
    raw.drop_channels(['Audio left', 'Audio right'])

    dig_montage = mne.channels.make_standard_montage("standard_1005")
    _ = raw.set_montage(dig_montage)

    if re_reference:
        logger.info(f"Loading data: {input_file}")
        raw.load_data()
        logger.info("Applying average reference to data")
        raw.set_eeg_reference("average")

    if input_file.name == "sub-michal_ses-01.vhdr":
        logger.info("Applying trigger correction to sub-michal_ses-01.vhdr, as this file was recorded using a "
                    "legacy trigger set")
        process_michal_ses_01(input_file, raw)

    logger.info("Saving")
    raw.save(output_file)


if __name__ == "__main__":
    app()
