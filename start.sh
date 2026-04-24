#!/bin/bash
source ~/Trading/kronos-trader/venv/bin/activate
cd ~/Trading/kronos-trader/backend
uvicorn main:app --host 0.0.0.0 --port 8000
