#!/bin/bash

# T-Detect Reproduction Script: HART Dataset
# This script reproduces the main results on the HART benchmark

set -e

echo "=========================================="
echo "T-Detect HART Dataset Reproduction"
echo "=========================================="

# Configuration
RESULT_DIR="./exp_reproduction_hart"
DATA_PATH="./benchmark/hart"
DATASETS="essay.dev,essay.test,news.dev,news.test,writing.dev,writing.test,arxiv.dev,arxiv.test"

# Create result directory
mkdir -p ${RESULT_DIR}

echo "Running T-Detect on HART datasets..."
echo "Datasets: ${DATASETS}"
echo "Result directory: ${RESULT_DIR}"
echo ""

# Run T-Detect experiments
echo "1. Running T-Detect (1D)..."
python scripts/delegate_detector.py \
    --data_path ${DATA_PATH} \
    --result_path ${RESULT_DIR} \
    --datasets ${DATASETS} \
    --detectors t_detect \
    --verbose

echo ""
echo "2. Running T-Detect with Content dimension (2D)..."
python scripts/delegate_detector.py \
    --data_path ${DATA_PATH} \
    --result_path ${RESULT_DIR} \
    --datasets ${DATASETS} \
    --detectors "C(t_detect)" \
    --verbose

echo ""
echo "3. Running T-Detect with Content+Text dimensions (CT)..."
python scripts/delegate_detector.py \
    --data_path ${DATA_PATH} \
    --result_path ${RESULT_DIR} \
    --datasets ${DATASETS} \
    --detectors "CT(t_detect)" \
    --verbose

echo ""
echo "4. Running baseline comparisons..."

# FastDetectGPT baseline
echo "  - FastDetectGPT baseline..."
python scripts/delegate_detector.py \
    --data_path ${DATA_PATH} \
    --result_path ${RESULT_DIR} \
    --datasets ${DATASETS} \
    --detectors fast_detect \
    --verbose

# Binoculars baseline
echo "  - Binoculars baseline..."
python scripts/delegate_detector.py \
    --data_path ${DATA_PATH} \
    --result_path ${RESULT_DIR} \
    --datasets ${DATASETS} \
    --detectors binoculars \
    --verbose

# CT variants of baselines
echo "  - CT(FastDetectGPT)..."
python scripts/delegate_detector.py \
    --data_path ${DATA_PATH} \
    --result_path ${RESULT_DIR} \
    --datasets ${DATASETS} \
    --detectors "CT(fast_detect)" \
    --verbose

echo "  - CT(Binoculars)..."
python scripts/delegate_detector.py \
    --data_path ${DATA_PATH} \
    --result_path ${RESULT_DIR} \
    --datasets ${DATASETS} \
    --detectors "CT(binoculars)" \
    --verbose
