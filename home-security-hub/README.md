# Home Security Hub

Starter project scaffold for a local-first home security pipeline with face/object perception, rule-based decisions, and pluggable outputs.

## Quick start (Conda)

```bash
conda env create -f environment.yml
conda activate hshub
cp .env.example .env
python -m hshub.app
```

## Update environment after dependency changes

```bash
conda env update -f environment.yml --prune
```

## Layout

See the directory structure in this repository for module boundaries.
