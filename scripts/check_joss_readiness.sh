#!/bin/bash
# JOSS Pre-Submission Validation Script

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║         JOSS PRE-SUBMISSION VALIDATION                       ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# Function to check file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
        ((ERRORS++))
    fi
}

# Function to check file contains text
check_content() {
    if grep -q "$2" "$1" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $3"
    else
        echo -e "${YELLOW}⚠${NC} $3"
        ((WARNINGS++))
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📄 REQUIRED FILES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

check_file "paper.md" "paper.md exists"
check_file "paper.bib" "paper.bib exists"
check_file "LICENSE" "LICENSE file exists"
check_file "README.md" "README.md exists"
check_file "CITATION.cff" "CITATION.cff exists"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 PAPER CONTENT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

check_content "paper.md" "^title:" "Title field present"
check_content "paper.md" "^tags:" "Tags field present"
check_content "paper.md" "^authors:" "Authors field present"
check_content "paper.md" "^date:" "Date field present"
check_content "paper.md" "^bibliography:" "Bibliography reference present"
check_content "paper.md" "# Summary" "Summary section present"
check_content "paper.md" "# Statement of Need" "Statement of Need section present"

# Check ORCID placeholder
if grep -q "0000-0000-0000-0000" "paper.md" 2>/dev/null; then
    echo -e "${YELLOW}⚠${NC} ORCID is placeholder - UPDATE REQUIRED"
    ((WARNINGS++))
else
    echo -e "${GREEN}✓${NC} ORCID appears to be set"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📚 REPOSITORY STRUCTURE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

check_file "setup.py" "setup.py exists (or pyproject.toml)"
check_file "requirements.txt" "requirements.txt exists"

# Check for examples
if [ -d "examples" ] && [ "$(ls -A examples/*.py 2>/dev/null)" ]; then
    echo -e "${GREEN}✓${NC} Examples directory with Python files"
else
    echo -e "${YELLOW}⚠${NC} Examples directory not found or empty"
    ((WARNINGS++))
fi

# Check for tests
if [ -d "tests" ] || [ -d "sofia/tests" ]; then
    echo -e "${GREEN}✓${NC} Tests directory exists"
else
    echo -e "${RED}✗${NC} Tests directory not found"
    ((ERRORS++))
fi

# Check for documentation
if [ -d "docs" ] && [ "$(ls -A docs/*.md 2>/dev/null)" ]; then
    echo -e "${GREEN}✓${NC} Documentation directory with markdown files"
else
    echo -e "${YELLOW}⚠${NC} Documentation directory not found or empty"
    ((WARNINGS++))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 GIT STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if git repository
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Git repository"
    
    # Check for uncommitted changes
    if [ -z "$(git status --porcelain)" ]; then
        echo -e "${GREEN}✓${NC} No uncommitted changes"
    else
        echo -e "${YELLOW}⚠${NC} Uncommitted changes present"
        ((WARNINGS++))
    fi
    
    # Check for remote
    if git remote -v | grep -q "github.com"; then
        echo -e "${GREEN}✓${NC} GitHub remote configured"
    else
        echo -e "${RED}✗${NC} No GitHub remote found"
        ((ERRORS++))
    fi
    
    # Check for tags
    if git tag | grep -q "v0.1.0"; then
        echo -e "${GREEN}✓${NC} Version tag v0.1.0 exists"
    else
        echo -e "${YELLOW}⚠${NC} Version tag v0.1.0 not found - create GitHub release"
        ((WARNINGS++))
    fi
else
    echo -e "${RED}✗${NC} Not a git repository"
    ((ERRORS++))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ PERFECT! All checks passed!${NC}"
    echo ""
    echo "You are ready to submit to JOSS! 🚀"
    echo ""
    echo "Next steps:"
    echo "  1. Update ORCID in paper.md if needed"
    echo "  2. Publish to PyPI: twine upload dist/*"
    echo "  3. Create GitHub release v0.1.0"
    echo "  4. Submit at: https://joss.theoj.org/papers/new"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ GOOD with warnings (${WARNINGS} warnings)${NC}"
    echo ""
    echo "You can submit, but consider addressing warnings first."
    echo "See JOSS_SUBMISSION_GUIDE.md for details."
else
    echo -e "${RED}✗ ISSUES FOUND (${ERRORS} errors, ${WARNINGS} warnings)${NC}"
    echo ""
    echo "Please fix errors before submitting to JOSS."
    echo "See JOSS_SUBMISSION_GUIDE.md for help."
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit $ERRORS
