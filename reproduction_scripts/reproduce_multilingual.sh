#!/bin/bash

# T-Detect Reproduction Script: Multilingual Evaluation
# This script reproduces the multilingual results across 5 languages

set -e

echo "=========================================="
echo "T-Detect Multilingual Reproduction"
echo "=========================================="

# Configuration
RESULT_DIR="./exp_reproduction_multilingual"
DATA_PATH="./benchmark/hart"

# Language datasets
LANGUAGES=("en" "es" "ar" "zh" "fr")
LANGUAGE_NAMES=("English" "Spanish" "Arabic" "Chinese" "French")

# Create result directory
mkdir -p ${RESULT_DIR}

echo "Running T-Detect on multilingual datasets..."
echo "Languages: ${LANGUAGE_NAMES[*]}"
echo "Result directory: ${RESULT_DIR}"
echo ""

# Function to run experiments for a specific language
run_language_experiments() {
    local lang_code=$1
    local lang_name=$2
    local datasets=""
    
    if [ "$lang_code" = "en" ]; then
        datasets="news.dev,news.test"
    else
        datasets="news-${lang_code}.dev,news-${lang_code}.test"
    fi
    
    echo "Running experiments for ${lang_name} (${lang_code})..."
    echo "Datasets: ${datasets}"
    
    # T-Detect
    echo "  - T-Detect..."
    python scripts/delegate_detector.py \
        --data_path ${DATA_PATH} \
        --result_path ${RESULT_DIR} \
        --datasets ${datasets} \
        --detectors t_detect \
        --verbose
    
    # CT(T-Detect)
    echo "  - CT(T-Detect)..."
    python scripts/delegate_detector.py \
        --data_path ${DATA_PATH} \
        --result_path ${RESULT_DIR} \
        --datasets ${datasets} \
        --detectors "CT(t_detect)" \
        --verbose
    
    # Baselines for comparison
    echo "  - FastDetectGPT baseline..."
    python scripts/delegate_detector.py \
        --data_path ${DATA_PATH} \
        --result_path ${RESULT_DIR} \
        --datasets ${datasets} \
        --detectors fast_detect \
        --verbose
    
    echo "  - Binoculars baseline..."
    python scripts/delegate_detector.py \
        --data_path ${DATA_PATH} \
        --result_path ${RESULT_DIR} \
        --datasets ${datasets} \
        --detectors binoculars \
        --verbose
    
    echo "  ${lang_name} experiments complete."
    echo ""
}

# Run experiments for each language
for i in "${!LANGUAGES[@]}"; do
    run_language_experiments "${LANGUAGES[$i]}" "${LANGUAGE_NAMES[$i]}"
done

