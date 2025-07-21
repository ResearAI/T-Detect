#!/bin/bash

# T-Detect Reproduction Script: Comprehensive Baseline Comparison
# This script runs all baseline methods for comprehensive comparison

set -e

echo "=========================================="
echo "T-Detect Comprehensive Baseline Reproduction"
echo "=========================================="

# Configuration
RESULT_DIR="./exp_reproduction_baselines"
HART_PATH="./benchmark/hart"
RAID_PATH="./benchmark/raid"

# Create result directory
mkdir -p ${RESULT_DIR}

echo "Running comprehensive baseline comparison..."
echo "Result directory: ${RESULT_DIR}"
echo ""

# HART dataset evaluation
echo "1. HART Dataset Evaluation"
echo "================================"

HART_DATASETS="essay.dev,essay.test,news.dev,news.test"

echo "Running all detectors on HART datasets..."

# Statistical methods (zero-shot)
echo "  Statistical Methods:"
for detector in "t_detect" "fast_detect" "binoculars" "log_perplexity" "log_rank" "lrr"; do
    echo "    - ${detector}..."
    python scripts/delegate_detector.py \
        --data_path ${HART_PATH} \
        --result_path ${RESULT_DIR} \
        --datasets ${HART_DATASETS} \
        --detectors ${detector} \
        --verbose
done

# Neural classifiers
echo "  Neural Classifiers:"
for detector in "roberta" "radar"; do
    echo "    - ${detector}..."
    python scripts/delegate_detector.py \
        --data_path ${HART_PATH} \
        --result_path ${RESULT_DIR} \
        --datasets ${HART_DATASETS} \
        --detectors ${detector} \
        --verbose
done

# Two-dimensional variants
echo "  Two-Dimensional Methods:"
for detector in "CT(t_detect)" "CT(fast_detect)" "CT(binoculars)"; do
    echo "    - ${detector}..."
    python scripts/delegate_detector.py \
        --data_path ${HART_PATH} \
        --result_path ${RESULT_DIR} \
        --datasets ${HART_DATASETS} \
        --detectors "${detector}" \
        --verbose
done

echo ""
echo "2. RAID Dataset Evaluation (Adversarial)"
echo "========================================"

RAID_DATASETS="raid.dev,raid.test"

echo "Running key detectors on RAID datasets..."

# Focus on most important methods for adversarial evaluation
for detector in "t_detect" "fast_detect" "fast_detect_raid" "binoculars" "CT(t_detect)" "CT(fast_detect)" "CT(binoculars)"; do
    echo "  - ${detector}..."
    python scripts/delegate_detector.py \
        --data_path ${RAID_PATH} \
        --result_path ${RESULT_DIR} \
        --datasets ${RAID_DATASETS} \
        --detectors "${detector}" \
        --verbose
done

echo ""
echo "3. Speed and Efficiency Benchmarking"
echo "==================================="

echo "Running timing comparison..."
python scripts/timing_comparison.py \
    --detectors t_detect,fast_detect,binoculars \
    --num_samples 100 \
    --output ${RESULT_DIR}/timing_results.json

echo ""
echo "=========================================="
echo "Comprehensive Baseline Reproduction Complete!"
echo "=========================================="
echo "Results saved to: ${RESULT_DIR}"
echo ""

