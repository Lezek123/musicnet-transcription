#!/usr/bin/env bash

SCRIPT_PATH="$(dirname $0)"
cd $SCRIPT_PATH

python ../../preprocessing/midi_to_wav/convert.py
python ../../preprocessing/wav_specs_and_notes/preprocess.py
python ./train.py