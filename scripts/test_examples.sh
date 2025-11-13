#!/bin/bash
#
# Test all SOFIA examples
# This script runs each example and reports success/failure
#
# Usage:
#   ./test_examples.sh              # Normal mode (quiet output)
#   ./test_examples.sh -v           # Verbose mode (show all output)
#   ./test_examples.sh --verbose    # Verbose mode (show all output)
#

set +e  # Don't exit on error

# Check for verbose mode
VERBOSE=0
if [[ "$1" == "-v" || "$1" == "--verbose" ]]; then
    VERBOSE=1
fi

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

# Examples that need special handling (longer timeout or special parameters)
# These anisotropic examples are computationally intensive, so we run them with fewer iterations for testing
# The --no-plot flag skips visualization for 2-3× speedup (visualization takes ~55% of runtime)
declare -A SPECIAL_EXAMPLES
SPECIAL_EXAMPLES["anisotropic_levelset_adaptation.py"]="--max-iter 1 --no-plot"
SPECIAL_EXAMPLES["anisotropic_remeshing_normalized.py"]="--max-iter 1 --target-complexity 150 --no-plot"

# Timeout for each example (in seconds)
# Reduced from 120s to 60s now that anisotropic examples use --no-plot
TIMEOUT=60

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
    
    # Check if this example needs special parameters
    special_args="${SPECIAL_EXAMPLES[$filename]}"
    
    # Run the example with a timeout
    if [ $VERBOSE -eq 1 ]; then
        echo ""  # New line before verbose output
        if [ -n "$special_args" ]; then
            echo "  Running: python $example $special_args"
            timeout ${TIMEOUT}s python "$example" $special_args
        else
            echo "  Running: python $example"
            timeout ${TIMEOUT}s python "$example"
        fi
    else
        if [ -n "$special_args" ]; then
            timeout ${TIMEOUT}s python "$example" $special_args > /dev/null 2>&1
        else
            timeout ${TIMEOUT}s python "$example" > /dev/null 2>&1
        fi
    fi
    
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ OK"
        ((SUCCESS++))
    elif [ $exit_code -eq 124 ]; then
        echo "✗ TIMEOUT (>${TIMEOUT}s)"
        ((FAILED++))
        FAILED_FILES+=("$filename (timeout)")
    else
        echo "✗ FAILED (exit code: $exit_code)"
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
