@echo off
cd /d "%~dp0program"
python -m pip install -r requirements.txt --quiet
python run_me.py
pause
