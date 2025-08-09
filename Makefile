#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = auditory-intention-decoding-data-analysis
PYTHON_VERSION = 3.8
PYTHON_INTERPRETER = poetry run python

FIGURES = "reports/figures"
SRC = "auditory_intention_decoding_data_analysis"
DUMMY_FILE = ".dummy"

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies

requirements:
	poetry install


.PHONY: clean
## Delete all compiled Python files

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

RAW_FILES := $(wildcard data/raw/sub-*/ses-*/*.vhdr)
RE_REFERENCED_FILES := $(foreach file, $(RAW_FILES), $(shell echo $(file) | sed 's|data/raw/sub-[^/]*/ses-[^/]*/\(.*\).vhdr|data/processed/\1-average_reference.fif|'))
PROCESSED_FILES := $(foreach file, $(RAW_FILES), $(shell echo $(file) | sed 's|data/raw/sub-[^/]*/ses-[^/]*/\(.*\).vhdr|data/processed/\1.fif|'))

## Make Dataset

data: requirements $(RE_REFERENCED_FILES) $(PROCESSED_FILES)
data/processed/%-average_reference.fif: data/raw/sub-*/ses-*/%.vhdr
	$(PYTHON_INTERPRETER) auditory_intention_decoding_data_analysis/dataset.py $< $@ true
data/processed/%.fif: data/raw/sub-*/ses-*/%.vhdr
	$(PYTHON_INTERPRETER) auditory_intention_decoding_data_analysis/dataset.py $< $@ false

#--------------------------------------------------------------------------------
ERP_PLOT_TARGET := "$(FIGURES)/erp-plots/$(DUMMY_FILE)"
ERP_PLOT_TARGET_PER_SUBJECT := $(PROCESSED_FILES:data/processed/%.fif=$(FIGURES)/erp-plots-per-subject/%/$(DUMMY_FILE))

## Make ERP-plots

erp-plot: $(ERP_PLOT_TARGET) $(ERP_PLOT_TARGET_PER_SUBJECT)
$(ERP_PLOT_TARGET): $(PROCESSED_FILES)
	$(PYTHON_INTERPRETER) $(SRC)/plots/plot_erp.py $^ $@
$(FIGURES)/erp-plots-per-subject/%/$(DUMMY_FILE): data/processed/%.fif
	$(PYTHON_INTERPRETER) $(SRC)/plots/plot_erp.py $^ $@
#--------------------------------------------------------------------------------

SAMPLES_TARGETS := $(PROCESSED_FILES:data/processed/%.fif=data/interim/samples/%-epo.fif)

## Build samples to be used by eegnet

build-samples: $(SAMPLES_TARGETS)
data/interim/samples/%-epo.fif: data/processed/%.fif
	mkdir -p $$(dirname $@)
	$(PYTHON_INTERPRETER) $(SRC)/models/build_samples.py $^ $@ --sfreq 128 --tmin -0.2 --tmax 1

#--------------------------------------------------------------------------------

SAMPLES_TF_TARGETS := $(PROCESSED_FILES:data/processed/%.fif=data/interim/samples-tf/%-epo.fif)

## Build samples to be used by the time-frequency analysis

build-samples-tf: $(SAMPLES_TF_TARGETS)
data/interim/samples-tf/%-epo.fif: data/processed/%.fif
	mkdir -p $$(dirname $@)
	$(PYTHON_INTERPRETER) $(SRC)/models/build_samples.py $^ $@ --tmin -0.2 --tmax 1

#--------------------------------------------------------------------------------


EEGNET_TARGETS := $(SAMPLES_TARGETS:data/interim/samples/%-epo.fif=reports/eegnet/%)
EEGNET_ALL := "reports/eegnet/all"
EEGNET_MICHAL := "reports/eegnet/all-michal"

## Train EEGNet

train-eegnet: $(EEGNET_TARGETS) $(EEGNET_ALL) $(EEGNET_MICHAL)
reports/eegnet/%: data/interim/samples/%-epo.fif
	mkdir -p $@
	$(PYTHON_INTERPRETER) $(SRC)/models/eegnet.py $^ $@ 200
	touch $@
$(EEGNET_ALL): $(SAMPLES_TARGETS)
	mkdir -p $@
	$(PYTHON_INTERPRETER) $(SRC)/models/eegnet.py $^ $@ 600
	touch $@
$(EEGNET_MICHAL): data/interim/samples/sub-michal_ses-01-epo.fif data/interim/samples/sub-michal_ses-02-epo.fif data/interim/samples/sub-michal_ses-03-epo.fif
	mkdir -p $@
	$(PYTHON_INTERPRETER) $(SRC)/models/eegnet.py $^ $@ 500
	touch $@


#--------------------------------------------------------------------------------


RIEMANN_TARGETS := $(SAMPLES_TARGETS:data/interim/samples/%-epo.fif=reports/riemann/%)
RIEMANN_ALL := reports/riemann/all
RIEMANN_MICHAL := reports/riemann/all-michal

## Train Riemann
train-riemann: $(RIEMANN_TARGETS) $(RIEMANN_ALL) $(RIEMANN_MICHAL)

reports/riemann/%: data/interim/samples/%-epo.fif
	mkdir -p $@
	$(PYTHON_INTERPRETER) $(SRC)/models/erp_classification.py $^ $@
	touch $@

$(RIEMANN_ALL): $(SAMPLES_TARGETS)
	mkdir -p $@
	$(PYTHON_INTERPRETER) $(SRC)/models/erp_classification.py $^ $@
	touch $@

$(RIEMANN_MICHAL): data/interim/samples/sub-michal_ses-01-epo.fif data/interim/samples/sub-michal_ses-02-epo.fif data/interim/samples/sub-michal_ses-03-epo.fif
	mkdir -p $@
	$(PYTHON_INTERPRETER) $(SRC)/models/erp_classification.py $^ $@
	touch $@

## Train Riemann Grid
#train-riemann-grid: $(RIEMANN_TARGETS) $(RIEMANN_ALL) $(RIEMANN_MICHAL)
#
#reports/riemann/%: data/interim/samples/%-epo.fif
#	mkdir -p $@
#	$(PYTHON_INTERPRETER) $(SRC)/models/erp_classification_grid_search.py $^ $@
#	touch $@
#
#$(RIEMANN_ALL): $(SAMPLES_TARGETS)
#	mkdir -p $@
#	$(PYTHON_INTERPRETER) $(SRC)/models/erp_classification_grid_search.py $^ $@
#	touch $@
#
#$(RIEMANN_MICHAL): data/interim/samples/sub-michal_ses-01-epo.fif data/interim/samples/sub-michal_ses-02-epo.fif data/interim/samples/sub-michal_ses-03-epo.fif
#	mkdir -p $@
#	$(PYTHON_INTERPRETER) $(SRC)/models/erp_classification_grid_search.py $^ $@
#	touch $@



## Train Riemann Grid Michal
train-riemann-grid-michal: $(RIEMANN_MICHAL)

$(RIEMANN_MICHAL): data/interim/samples/sub-michal_ses-01-epo.fif data/interim/samples/sub-michal_ses-02-epo.fif data/interim/samples/sub-michal_ses-03-epo.fif
	mkdir -p $@
	$(PYTHON_INTERPRETER) $(SRC)/models/erp_classification_grid_search.py $^ $@
	touch $@



#--------------------------------------------------------------------------------

PERMUTATION_ANALYSIS_TARGETS := $(EEGNET_TARGETS:reports/eegnet/%=reports/permutation-analysis/%.log)
PERMUTATION_ANALYSIS_TARGETS += reports/permutation-analysis/all.log
PERMUTATION_ANALYSIS_TARGETS += reports/permutation-analysis/all-michal.log

## Permutation analysis

permutation-analysis: $(PERMUTATION_ANALYSIS_TARGETS)
reports/permutation-analysis/%.log: reports/eegnet/%/prediction.pkl
	mkdir -p $$(dirname $@)
	$(PYTHON_INTERPRETER) $(SRC)/models/permutation_analysis.py $^ | tee $@


#--------------------------------------------------------------------------------

TIME_FREQUENCY_TARGETS := $(SAMPLES_TF_TARGETS:data/interim/samples-tf/%-epo.fif=reports/figures/time-frequency/%)

## Time-frequency plots
time-frequency-analysis: $(TIME_FREQUENCY_TARGETS)
reports/figures/time-frequency/%: data/interim/samples-tf/%-epo.fif
	mkdir -p $$(dirname $@)
	$(PYTHON_INTERPRETER) $(SRC)/plots/time_frequency.py $^ $@
	touch $@


#--------------------------------------------------------------------------------


## Make all rules

all: data erp-plot build-samples train-eegnet permutation-analysis time-frequency-analysis


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
