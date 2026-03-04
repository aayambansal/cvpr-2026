#!/bin/bash
# Download results from Modal volume and run analysis
# Usage: bash scripts/download_and_analyze.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Checking Modal volume for results ==="
modal volume ls repnas-results /

echo ""
echo "=== Downloading results ==="

# Try downloading upgrades
if modal volume get repnas-results repnas_upgrades.json experiments/repnas_upgrades.json 2>/dev/null; then
    echo "✓ Downloaded repnas_upgrades.json"
else
    echo "✗ repnas_upgrades.json not available yet"
fi

# Try downloading partial upgrades
if modal volume get repnas-results repnas_upgrades_partial.json experiments/repnas_upgrades_partial.json 2>/dev/null; then
    echo "✓ Downloaded repnas_upgrades_partial.json"
else
    echo "  (no partial results either)"
fi

# Try downloading transfer
if modal volume get repnas-results repnas_transfer.json experiments/repnas_transfer.json 2>/dev/null; then
    echo "✓ Downloaded repnas_transfer.json"
else
    echo "✗ repnas_transfer.json not available yet"
fi

echo ""
echo "=== Local experiment files ==="
ls -la experiments/*.json

echo ""
echo "=== Running analysis ==="
python3 scripts/analyze_new_experiments.py
