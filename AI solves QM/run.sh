#!/usr/bin/env bash
cd "$(dirname "$0")/program"
python3 -m pip install -r requirements.txt --quiet
python3 run_me.py
