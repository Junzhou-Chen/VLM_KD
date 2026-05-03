#!/bin/bash
conda create -n VLM_KD python=3.11
pip install -r ../requirements.txt
conda install -c conda-forge ffmpeg
