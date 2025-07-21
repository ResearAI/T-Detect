#!/bin/bash

# T-Detect Reproduction Script: RAID Dataset (Adversarial Evaluation)
# This script reproduces the adversarial robustness results on the RAID benchmark

set -e

echo "=========================================="
echo "T-Detect RAID Dataset Reproduction"
echo "=========================================="

# Configuration
RESULT_DIR="./exp_reproduction_raid"
DATA_PATH="./benchmark/raid"
DATASETS="raid.dev,raid.test,nonnative.test"

# Create result directory
mkdir -p ${RESULT_DIR}

echo "Running T-Detect on RAID datasets (adversarial evaluation)..."
echo "Datasets: ${DATASETS}"
echo "Result directory: ${RESULT_DIR}"
echo ""

# Run T-Detect experiments
echo "1. Running T-Detect (1D) on adversarial text..."
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
echo "3. Running T-Detect with Content+Text dimensions (CT) - BEST PERFORMANCE..."
python scripts/delegate_detector.py \
    --data_path ${DATA_PATH} \
    --result_path ${RESULT_DIR} \
    --datasets ${DATASETS} \
    --detectors "CT(t_detect)" \
    --verbose

echo ""
echo "4. Running baseline comparisons on adversarial text..."

# FastDetectGPT baseline
echo "  - FastDetectGPT baseline..."
python scripts/delegate_detector.py \
    --data_path ${DATA_PATH} \
    --result_path ${RESULT_DIR} \
    --datasets ${DATASETS} \
    --detectors fast_detect \
    --verbose

# FastDetectGPT RAID variant
echo "  - FastDetectGPT RAID variant..."
python scripts/delegate_detector.py \
    --data_path ${DATA_PATH} \
    --result_path ${RESULT_DIR} \
    --datasets ${DATASETS} \
    --detectors fast_detect_raid \
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

echo ""
echo "5. Additional robustness evaluation..."

# Neural classifier baselines for comparison
echo "  - RoBERTa classifier..."
python scripts/delegate_detector.py \
    --data_path ${DATA_PATH} \
    --result_path ${RESULT_DIR} \
    --datasets ${DATASETS} \
    --detectors roberta \
    --verbose

