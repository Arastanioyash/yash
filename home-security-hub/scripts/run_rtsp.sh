#!/usr/bin/env bash
set -euo pipefail
export HSHUB_CONFIG=configs/prod.yaml
python -m hshub.app
