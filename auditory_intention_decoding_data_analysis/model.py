import functools
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from typing import List, Optional, Tuple

import mne
import numpy as np


class ETagger(Enum):
    RAW = "RawTagger"
    AM = "AMTagger"
    FM = "FMTagger"
    FLIPPED_FM = "FlippedFMTagger"
    NOISE_TAGGING = "NoiseTaggingTagger"
    SHIFT_SUM = "ShiftSumTagger"
    SPECTRUM_SHIFT = "SpectrumShiftTagger"
    BINAURAL = "BinauralTagger"

    @staticmethod
    def get_tagger(name: str) -> "ETagger":
        for ele in ETagger:
            if ele.value == name:
                return ele

        raise ValueError(f"The provided name was not found among the taggers! name: {name}")

    def __eq__(self, other: "ETagger") -> bool:
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)


class Triggers:
    NEW_PROMPT = 100
    NEW_STIMULUS = 101
    END_STIMULUS = 102
    ATTENTION_CHECK_ACTION = 120
    RESTING_STATE_EYES_OPEN_INTRODUCTION = 20
    RESTING_STATE_EYES_OPEN = 21
    RESTING_STATE_EYES_CLOSED_INTRODUCTION = 22
    RESTING_STATE_EYES_CLOSED = 23
    INTRODUCTION = 1
    EXPERIMENT_INTRODUCTION = 2
    EXPERIMENT = 3
    BREAK_START = 4
    OUTRO = 5
    EXAMPLE = 6
    EXAMPLE_INTRODUCTION = 7
    ATTENTION_CHECK = 8
    INACTIVE = 200

    OPTION_START_LEGACY = 111
    TARGET_START_LEGACY = 113
    OPTION_END_LEGACY = 112
    TARGET_END_LEGACY = 114

    OPTION_START_OFFSET = 50
    TARGET_START_OFFSET = 70
    OPTION_END = 69
    TARGET_END = 89


TRIGGER_NAMES = {
    Triggers.NEW_PROMPT:                             "NEW PROMPT",
    Triggers.NEW_STIMULUS:                           "NEW STIMULUS",
    Triggers.END_STIMULUS:                           "END STIMULUS",
    Triggers.ATTENTION_CHECK_ACTION:                 "ATTENTION_CHECK_ACTION",
    Triggers.RESTING_STATE_EYES_OPEN_INTRODUCTION:   "RESTING_STATE_EYES_OPEN_INTRODUCTION",
    Triggers.RESTING_STATE_EYES_OPEN:                "RESTING_STATE_EYES_OPEN",
    Triggers.RESTING_STATE_EYES_CLOSED_INTRODUCTION: "RESTING_STATE_EYES_CLOSED_INTRODUCTION",
    Triggers.RESTING_STATE_EYES_CLOSED:              "RESTING_STATE_EYES_CLOSED",
    Triggers.INTRODUCTION:                           "INTRODUCTION",
    Triggers.EXPERIMENT_INTRODUCTION:                "EXPERIMENT_INTRODUCTION",
    Triggers.EXPERIMENT:                             "EXPERIMENT",
    Triggers.BREAK_START:                            "BREAK_START",
    Triggers.OUTRO:                                  "OUTRO",
    Triggers.EXAMPLE:                                "EXAMPLE",
    Triggers.EXAMPLE_INTRODUCTION:                   "EXAMPLE_INTRODUCTION",
    Triggers.ATTENTION_CHECK:                        "ATTENTION_CHECK",
    Triggers.INACTIVE:                               "INACTIVE",

    Triggers.OPTION_START_LEGACY:                    "OPTION_START_LEGACY",
    Triggers.TARGET_START_LEGACY:                    "TARGET_START_LEGACY",
    Triggers.OPTION_END_LEGACY:                      "OPTION END_LEGACY",
    Triggers.TARGET_END_LEGACY:                      "TARGET_END_LEGACY",

    Triggers.OPTION_START_OFFSET:                    "OPTION_START",
    Triggers.TARGET_START_OFFSET:                    "TARGET_START",
    Triggers.OPTION_END:                             "OPTION_END",
    Triggers.TARGET_END:                             "TARGET_END"
}

TRIGGER_OFFSETS = [
    ETagger.RAW,
    ETagger.AM,
    ETagger.FM,
    ETagger.FLIPPED_FM,
    ETagger.SPECTRUM_SHIFT,
    ETagger.SHIFT_SUM,
    ETagger.BINAURAL,
    ETagger.NOISE_TAGGING
]


def get_trigger_name(trigger: int) -> str:
    if Triggers.OPTION_START_OFFSET <= trigger < Triggers.OPTION_END:
        offset = trigger - Triggers.OPTION_START_OFFSET

        # NEW PROMPT_2
        return TRIGGER_NAMES[Triggers.OPTION_START_OFFSET] + "_" + TRIGGER_OFFSETS[offset].value

    elif Triggers.TARGET_START_OFFSET <= trigger < Triggers.TARGET_END:
        offset = trigger - Triggers.TARGET_START_OFFSET

        return TRIGGER_NAMES[Triggers.TARGET_START_OFFSET] + "_" + TRIGGER_OFFSETS[offset].value

    else:
        return TRIGGER_NAMES[trigger]


def get_option_start_trigger(tagger: ETagger) -> int:
    return Triggers.OPTION_START_OFFSET + TRIGGER_OFFSETS.index(tagger)


def get_target_start_trigger(tagger: ETagger) -> int:
    return Triggers.TARGET_START_OFFSET + TRIGGER_OFFSETS.index(tagger)


@dataclass
class Data:
    raw: mne.io.Raw
    ch_names: List[str]
    fs: int
    triggers: List[Tuple[int, int]]

    @functools.cached_property
    def array(self) -> np.array:
        return self.raw.get_data()


@dataclass(frozen=True)
class Sample:
    tagger: ETagger
    interval: Tuple[int, int]
    audio_file: PathLike
    i_target: Optional[int]
    primer: int
    option_texts: List[str]
    option_intervals: List[Tuple[int, int]]

    def __post_init__(self) -> None:
        if self.interval[0] >= self.interval[1]:
            raise ValueError("Must supply a valid interval!")

        if any(interval[0] >= interval[1] for interval in self.option_intervals):
            raise ValueError("All of the option_intervals must be valid intervals!")

        if len(self.option_texts) != len(self.option_intervals):
            raise ValueError(f"There must be the same number of  elements in `option_text` and"
                             f"`option_slices` each! Was: {len(self.option_texts)} vs. {len(self.option_intervals)}")

        if self.i_target is not None and 0 > self.i_target:
            raise ValueError("`target` is not a valid index!")

        if 0 > self.primer:
            raise ValueError("`primer` is not in the valid range")

    @property
    def is_attention_check(self) -> bool:
        return self.i_target is None
