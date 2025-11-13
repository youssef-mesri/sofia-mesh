#!/usr/bin/env python3
"""
Pre-publication verification script for SOFIA

This script checks that all necessary files are in place and properly configured
before publishing to GitHub and PyPI.
"""

import os
import sys
import subprocess
from pathlib import Path

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check(condition, message, warning=False):
    """Print a check result"""
    if condition:
        print(f"{GREEN}OK{RESET} {message}")
        return True
    else:
        color = YELLOW if warning else RED
        symbol = '!' if warning else 'KO'
        print(f"{color}{symbol}{RESET} {message}")
        return not warning

def main():
    """Run all checks"""
    print("=" * 70)
    print(f"{BLUE}SOFIA Publication Verification{RESET}")
    print("=" * 70)
    print()
    
    all_passed = True
    warnings = []
    
    # Change to publication directory (parent of scripts/)
    pub_dir = Path(__file__).parent.parent
    os.chdir(pub_dir)
    
    # 1. Essential Files
    print(f"{BLUE}[1] Essential Files{RESET}")
    all_passed &= check(Path("LICENSE").exists(), "LICENSE file exists")
    all_passed &= check(Path("README.md").exists(), "README.md exists")
    all_passed &= check(Path("CITATION.cff").exists(), "CITATION.cff exists")
    all_passed &= check(Path("docs/CITATION.md").exists(), "docs/CITATION.md exists")
    all_passed &= check(Path("docs/CODE_OF_CONDUCT.md").exists(), "docs/CODE_OF_CONDUCT.md exists")
    all_passed &= check(Path("docs/CONTRIBUTING.md").exists(), "docs/CONTRIBUTING.md exists")
    all_passed &= check(Path("pyproject.toml").exists(), "pyproject.toml exists")
    all_passed &= check(Path("setup.py").exists(), "setup.py exists")
    print()
    
    # 2. Examples
    print(f"{BLUE}[2] Examples{RESET}")
    examples_dir = Path("examples")
    expected_examples = [
        "basic_remeshing.py",
        "quality_improvement.py",
        "boundary_operations.py",
        "adaptive_refinement.py",
        "mesh_coarsening.py",
        "mesh_workflow.py",
        "boundary_refinement.py",
        "combined_refinement.py",
    ]
    
    for example in expected_examples:
        all_passed &= check(
            (examples_dir / example).exists(),
            f"Example exists: {example}"
        )
    
    all_passed &= check(
        (examples_dir / "README.md").exists(),
        "Examples README.md exists"
    )
    print()
    
    # 3. Package Structure
    print(f"{BLUE}[3] Package Structure{RESET}")
    sofia_dir = Path("sofia")
    all_passed &= check(sofia_dir.exists() and sofia_dir.is_dir(), "sofia/ directory exists")
    all_passed &= check(
        (sofia_dir / "__init__.py").exists(),
        "sofia/__init__.py exists"
    )
    all_passed &= check(
        (sofia_dir / "core").exists(),
        "sofia/core/ directory exists"
    )
    print()
    
    # 4. Tests
    print(f"{BLUE}[4] Tests{RESET}")
    tests_dir = Path("tests")
    all_passed &= check(tests_dir.exists() and tests_dir.is_dir(), "tests/ directory exists")
    
    test_files = list(tests_dir.glob("test_*.py"))
    all_passed &= check(
        len(test_files) > 0,
        f"Found {len(test_files)} test files"
    )
    print()
    
    # 5. GitHub Configuration
    print(f"{BLUE}[5] GitHub Configuration{RESET}")
    gh_dir = Path(".github")
    all_passed &= check(gh_dir.exists(), ".github/ directory exists")
    all_passed &= check(
        (gh_dir / "workflows").exists(),
        ".github/workflows/ exists"
    )
    
    workflow_files = list((gh_dir / "workflows").glob("*.yml"))
    all_passed &= check(
        len(workflow_files) > 0,
        f"Found {len(workflow_files)} GitHub Actions workflows"
    )
    print()
    
    # 6. README Content Checks
    print(f"{BLUE}[6] README Content{RESET}")
    readme_path = Path("README.md")
    if readme_path.exists():
        readme_content = readme_path.read_text()
        
        check(
            "SOFIA" in readme_content,
            "README contains 'SOFIA'"
        )
        check(
            "pip install" in readme_content,
            "README contains installation instructions"
        )
        check(
            "import" in readme_content,
            "README contains code examples"
        )
        
        # Warnings for placeholder text
        if "your.email" in readme_content.lower():
            warnings.append("README contains placeholder email")
            check(False, "README email updated (found 'your.email')", warning=True)
        else:
            check(True, "README email appears updated")
        
        if "xxxx" in readme_content.lower():
            warnings.append("README may contain placeholder ORCID")
            check(False, "README ORCID updated (found 'XXXX')", warning=True)
        else:
            check(True, "README ORCID appears updated")
    print()
    
    # 7. pyproject.toml Configuration
    print(f"{BLUE}[7] Package Configuration{RESET}")
    pyproject_path = Path("pyproject.toml")
    if pyproject_path.exists():
        pyproject_content = pyproject_path.read_text()
        
        check(
            'name = "sofia-mesh"' in pyproject_content,
            "Package name is 'sofia-mesh'"
        )
        check(
            'version = ' in pyproject_content,
            "Package version is set"
        )
        check(
            'authors = ' in pyproject_content,
            "Package authors are set"
        )
        check(
            'dependencies = ' in pyproject_content,
            "Package dependencies are listed"
        )
    print()
    
    # 8. Git Status
    print(f"{BLUE}[8] Git Status{RESET}")
    try:
        # Check if we're in a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            check(True, "In a git repository")
            
            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.stdout.strip():
                warnings.append("Uncommitted changes detected")
                check(False, "No uncommitted changes", warning=True)
                print(f"   {YELLOW}Uncommitted files:{RESET}")
                lines = result.stdout.strip().split('\n')
                for line in lines[:5]:
                    print(f"   {line}")
                if len(lines) > 5:
                    remaining = len(lines) - 5
                    print(f"   ... and {remaining} more")
            else:
                check(True, "No uncommitted changes")
        else:
            warnings.append("Not in a git repository")
            check(False, "In a git repository", warning=True)
    except Exception as e:
        warnings.append(f"Git check failed: {e}")
        check(False, f"Git check failed: {e}", warning=True)
    print()
    
    # 9. Dependencies Check
    print(f"{BLUE}[9] Dependencies{RESET}")
    required_packages = ["numpy", "scipy", "matplotlib"]
    
    for package in required_packages:
        try:
            __import__(package)
            check(True, f"{package} is installed")
        except ImportError:
            warnings.append(f"{package} not installed")
            check(False, f"{package} is installed", warning=True)
    print()
    
    # 10. Example Execution (optional, can be slow)
    print(f"{BLUE}[10] Example Syntax Check{RESET}")
    
    for example in expected_examples[:3]:  # Check first 3 examples only
        example_path = examples_dir / example
        if example_path.exists():
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(example_path)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False
                )
                check(
                    result.returncode == 0,
                    f"{example} syntax is valid"
                )
            except Exception as e:
                warnings.append(f"Could not check {example}: {e}")
                check(False, f"Could not check {example}", warning=True)
    print()
    
    # Summary
    print("=" * 70)
    if all_passed:
        print(f"{GREEN}OK All critical checks passed!{RESET}")
    else:
        print(f"{RED}KO Some critical checks failed{RESET}")
    
    if warnings:
        print(f"{YELLOW}! {len(warnings)} warning(s):{RESET}")
        for warning in warnings:
            print(f"  • {warning}")
    
    print("=" * 70)
    print()
    
    if all_passed and not warnings:
        print(f"{GREEN} Ready for publication!{RESET}")
        print()
        print("Next steps:")
        print("  1. Review PUBLICATION_GUIDE.md")
        print("  2. Update README.md with your ORCID (if not done)")
        print("  3. Test on TestPyPI first")
        print("  4. Publish to GitHub")
        print("  5. Publish to PyPI")
        return 0
    elif all_passed:
        print(f"{YELLOW}! Ready with warnings{RESET}")
        print()
        print("Please review the warnings above before publishing.")
        print("See PUBLICATION_GUIDE.md for details.")
        return 0
    else:
        print(f"{RED}KO Not ready for publication{RESET}")
        print()
        print("Please fix the issues above before publishing.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
