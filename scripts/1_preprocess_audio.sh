#!/usr/bin/env bash
set -e 


python -m feature_extract.daic_audio_trim

python -m feature_extract.remove_ellie