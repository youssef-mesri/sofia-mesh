#!/bin/bash
#
# SOFIA Publication Cleanup - Automated Actions
# This script performs the high-priority cleanup tasks
#

set -e  # Exit on error

cd "$(dirname "$0")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         SOFIA Publication Cleanup - Automated Actions         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Phase 1: Move PNG files
echo -e "${BLUE}[1/6] Moving PNG visualizations...${NC}"
if ls *.png 1> /dev/null 2>&1; then
    mkdir -p examples/visualizations
    count=$(ls *.png 2>/dev/null | wc -l)
    mv *.png examples/visualizations/
    echo -e "${GREEN}OK Moved $count PNG files to examples/visualizations/${NC}"
else
    echo -e "${GREEN}OK No PNG files at root (already moved)${NC}"
fi
echo ""

# Phase 2: Check config files
echo -e "${BLUE}[2/6] Analyzing config files...${NC}"
if ls *.json 1> /dev/null 2>&1; then
    echo "Found JSON files at root:"
    ls -1 *.json
    echo ""
    echo "Checking if they're used by demos or tests..."
    
    for json in *.json; do
        echo -n "  $json: "
        usage=$(grep -r "$json" demos/ tests/ 2>/dev/null | wc -l)
        if [ $usage -gt 0 ]; then
            echo -e "${YELLOW}Used in $usage place(s) - keeping at root${NC}"
        else
            echo -e "${GREEN}Not used - can be moved to configs/${NC}"
            mkdir -p configs
            # mv "$json" configs/  # Commented out for safety - review first
        fi
    done
else
    echo -e "${GREEN}OK No JSON config files at root${NC}"
fi
echo ""

# Phase 3: Update .gitignore
echo -e "${BLUE}[3/6] Updating .gitignore...${NC}"
if ! grep -q "examples/visualizations/\*.png" .gitignore 2>/dev/null; then
    cat >> .gitignore << 'EOF'

# Example visualizations (generated)
examples/visualizations/*.png
*_result.png

# Build artifacts
dist/
dist-test/
build/
*.egg-info/

# IDE
.vscode/
.idea/

EOF
    echo -e "${GREEN}OK Updated .gitignore${NC}"
else
    echo -e "${GREEN}OK .gitignore already up to date${NC}"
fi
echo ""

# Phase 4: Check README ORCID
echo -e "${BLUE}[4/6] Checking README for placeholders...${NC}"
if grep -q "XXXX" README.md; then
    line=$(grep -n "XXXX" README.md | cut -d: -f1)
    echo -e "${YELLOW}WARNING Found placeholder at line $line${NC}"
    echo -e "${YELLOW}  Manual action needed: Edit README.md to fix ORCID${NC}"
    echo ""
    echo "Options:"
    echo "  1. Replace with real ORCID (if available)"
    echo "  2. Remove the ORCID line"
    echo "  3. Make it optional/generic"
    echo ""
    echo "To remove the line:"
    echo "  sed -i '/ORCID.*XXXX/d' README.md"
else
    echo -e "${GREEN}OK No placeholders found in README${NC}"
fi
echo ""

# Phase 5: Run verification
echo -e "${BLUE}[5/6] Running publication verification...${NC}"
if [ -f scripts/verify_publication.py ]; then
    python scripts/verify_publication.py > /tmp/verify_output.txt 2>&1
    
    # Check for warnings
    warnings=$(grep -c "warning" /tmp/verify_output.txt || true)
    
    if [ $warnings -eq 0 ]; then
        echo -e "${GREEN}OK All verification checks passed!${NC}"
    else
        echo -e "${YELLOW}WARNING Verification completed with $warnings warning(s)${NC}"
        echo "See full output:"
        echo "  cat /tmp/verify_output.txt"
    fi
else
    echo -e "${RED}ERROR Verification script not found${NC}"
fi
echo ""

# Phase 6: Git status summary
echo -e "${BLUE}[6/6] Git status summary...${NC}"
uncommitted=$(git status --short | wc -l)
if [ $uncommitted -eq 0 ]; then
    echo -e "${GREEN}OK No uncommitted changes${NC}"
else
    echo -e "${YELLOW}WARNING $uncommitted files with changes:${NC}"
    git status --short | head -10
    if [ $uncommitted -gt 10 ]; then
        echo "  ... and $((uncommitted - 10)) more"
    fi
    echo ""
    echo "To review all changes:"
    echo "  git status"
    echo ""
    echo "To commit all changes:"
    echo "  git add -A"
    echo "  git commit -m 'Final cleanup for publication'"
fi
echo ""

# Summary
echo "════════════════════════════════════════════════════════════════"
echo -e "${BLUE}Summary${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [ -f /tmp/verify_output.txt ]; then
    if grep -q "All critical checks passed" /tmp/verify_output.txt; then
        echo -e "${GREEN}OK Critical checks: PASSED${NC}"
    else
        echo -e "${RED}ERROR Critical checks: FAILED${NC}"
    fi
    
    warning_count=$(grep "warning(s):" /tmp/verify_output.txt | sed 's/.*(\([0-9]*\) warning.*/\1/' || echo "0")
    if [ "$warning_count" = "0" ]; then
        echo -e "${GREEN}OK Warnings: 0${NC}"
    else
        echo -e "${YELLOW}WARNING Warnings: $warning_count${NC}"
    fi
fi

echo ""
echo "Next steps:"
echo "  1. ${YELLOW}[MANUAL]${NC} Fix README ORCID placeholder (if present)"
echo "  2. Review: git status"
echo "  3. Commit: git add -A && git commit -m 'chore: Final cleanup'"
echo "  4. Test: ./scripts/test_examples.sh"
echo "  5. Build: python -m build"
echo ""
echo "For detailed guidance, see:"
echo "  • PUBLICATION_READINESS_ANALYSIS.md"
echo "  • docs/PUBLICATION_GUIDE.md"
echo ""
echo "════════════════════════════════════════════════════════════════"
