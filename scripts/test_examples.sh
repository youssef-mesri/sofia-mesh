#!/bin/bash
#
# Test all SOFIA examples
# This script runs each example and reports success/failure
#

set +e  # Don't exit on error

# Go to project root (parent of scripts/)
cd "$(dirname "$0")/.."

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              Testing SOFIA Examples                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

EXAMPLES_DIR="examples"
SUCCESS=0
FAILED=0
FAILED_FILES=()

# Check if examples directory exists
if [ ! -d "$EXAMPLES_DIR" ]; then
    echo "[ERROR] $EXAMPLES_DIR directory not found"
    exit 1
fi

# Find all Python example files
EXAMPLES=($(find "$EXAMPLES_DIR" -maxdepth 1 -name "*.py" | sort))

if [ ${#EXAMPLES[@]} -eq 0 ]; then
    echo " [ERROR] No example files found in $EXAMPLES_DIR"
    exit 1
fi

echo "Found ${#EXAMPLES[@]} example(s) to test"
echo ""

# Test each example
for example in "${EXAMPLES[@]}"; do
    filename=$(basename "$example")
    echo -n "Testing $filename... "
    
    # Run the example with a timeout
    timeout 60s python "$example" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "OK"
        ((SUCCESS++))
    elif [ $? -eq 124 ]; then
        echo " TIMEOUT (>60s)"
        ((FAILED++))
        FAILED_FILES+=("$filename (timeout)")
    else
        echo " FAILED"
        ((FAILED++))
        FAILED_FILES+=("$filename")
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Results:"
echo "  Passed: $SUCCESS"
echo "  Failed: $FAILED"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed examples:"
    for failed in "${FAILED_FILES[@]}"; do
        echo "  • $failed"
    done
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "[ERROR] Some examples failed"
    exit 1
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "[SUCCESS] All examples passed!"
    exit 0
fi
