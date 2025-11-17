# JOSS Submission Guide for SOFIA

This guide walks you through submitting SOFIA to the Journal of Open Source Software (JOSS).

## Prerequisites Checklist

Before submitting, ensure you have:

### Repository Requirements
- [x] **Public GitHub repository**: https://github.com/youssef-mesri/sofia
- [x] **Open Source License**: MIT License present
- [x] **README.md**: Comprehensive documentation
- [x] **CITATION.cff**: Citation file present
- [x] **Examples**: 8 complete examples with visualizations
- [x] **Tests**: 50+ unit tests
- [x] **Documentation**: Complete docs/ directory

### Paper Requirements
- [x] **paper.md**: JOSS article (created)
- [x] **paper.bib**: Bibliography (created)
- [x] **Proper formatting**: YAML header with metadata
- [x] **Word count**: ~1000-2500 words (target met)
- [x] **References**: Relevant citations included

### Pre-Submission Tasks

1. **Update ORCID** (Important!)
   - Register at https://orcid.org/ if you don't have an ORCID
   - Update your ORCID in:
     * `paper.md` (line 9): Replace `0000-0000-0000-0000`
     * `CITATION.cff` (line 18): Replace placeholder ORCID

2. **Verify PyPI Publication**
   - Publish to PyPI first: `twine upload dist/*`
   - Verify installation works: `pip install sofia-mesh`
   - This is required by JOSS before review starts

3. **Create GitHub Release**
   - Go to: https://github.com/youssef-mesri/sofia/releases/new
   - Tag: `v0.1.0`
   - Title: "SOFIA v0.1.0 - Initial Public Release"
   - Include release notes

4. **Check Paper Compilation**
   - Install JOSS PDF generator (optional, but helpful):
     ```bash
     docker pull openjournals/inara
     docker run --rm -v $PWD:/data openjournals/inara -o pdf,crossref paper.md
     ```
   - This generates `paper.pdf` to preview your submission

## Submission Steps

### Step 1: Register on JOSS

1. Go to https://joss.theoj.org/
2. Sign in with your GitHub account
3. JOSS will request access to your public repositories

### Step 2: Start Submission

1. Click **"Submit"** button on JOSS homepage
2. You'll be directed to https://joss.theoj.org/papers/new

### Step 3: Fill Submission Form

**Repository Information:**
- **Repository URL**: `https://github.com/youssef-mesri/sofia`
- **Version**: `v0.1.0` (the release tag you created)
- **Software Archive DOI**: Leave blank initially (Zenodo DOI will be generated)

**Paper Information:**
- **Branch**: `main` (or the branch containing paper.md)
- **Paper path**: `paper.md` (if at repository root)

**Categories:**
Select relevant categories:
- [ ] Computational Geometry
- [ ] Numerical Methods
- [ ] Scientific Computing
- [ ] Visualization

**Keywords** (select/add):
- mesh generation
- triangular mesh
- computational geometry
- mesh adaptation
- finite element method

### Step 4: Submit and Wait

1. Review all information carefully
2. Click **"Submit for review"**
3. JOSS bot will perform automated checks:
   - Repository accessibility
   - License detection
   - Paper compilation
   - Reference formatting
   - CITATION.cff validation

### Step 5: Address Bot Feedback

The JOSS bot (@whedon) will comment on your submission with:
- Green checks for passed requirements
- Warnings for recommendations
- Red X for issues that must be fixed

Common issues and fixes:
- **License not detected**: Ensure LICENSE file is in repository root
- **Paper won't compile**: Check YAML formatting and BibTeX references
- **Missing statement of need**: Ensure paper clearly explains why software is needed

### Step 6: Review Process

Once bot checks pass:
1. **Editor Assignment** (~1-2 weeks): An editor will be assigned
2. **Reviewer Selection** (~1-2 weeks): Editor selects 2 reviewers
3. **Review** (~2-4 weeks): Reviewers provide feedback via GitHub issues
4. **Revisions**: Address reviewer comments (usually minor)
5. **Acceptance**: Once reviewers approve, paper is published

**Total Timeline**: 1-3 months typical

## Common Reviewer Requests

Be prepared to address:

1. **Documentation improvements**
   - API documentation completeness
   - Example clarity
   - Installation instructions

2. **Testing coverage**
   - Edge cases coverage
   - Integration tests
   - Continuous Integration setup

3. **Paper clarity**
   - Statement of need strengthening
   - Comparison with similar tools
   - Performance benchmarks

4. **Code quality**
   - Type hints
   - Docstring completeness
   - Code organization

## Tips for Success

### Before Submission
- Test installation from PyPI on a clean environment
- Run all examples to ensure they work
- Verify all tests pass: `pytest tests/`
- Check documentation links work
- Proofread paper.md carefully

### During Review
- Respond to reviewer comments promptly
- Be open to suggestions
- Update paper.md based on feedback
- Tag new versions when making substantial changes
- Keep discussion professional and constructive

### Communication with Reviewers
Use GitHub issue comments to:
- Thank reviewers for feedback
- Explain changes you've made
- Ask for clarification if needed
- Use `@whedon` commands (e.g., `@whedon generate pdf` to regenerate paper)

## Useful @whedon Commands

Once your submission is active, you can use these commands in issue comments:

- `@whedon commands` - List all available commands
- `@whedon generate pdf` - Regenerate paper PDF
- `@whedon set version v0.1.0` - Update version
- `@whedon set archive <DOI>` - Set Zenodo DOI

## After Acceptance

Once accepted, JOSS will:
1. **Publish your paper** with a DOI
2. **Create Zenodo archive** (if not already done)
3. **Mint CrossRef DOI** for citations
4. **Add to JOSS website** with PDF

Your paper will be:
- Indexed in Google Scholar
- Citable with proper DOI
- Listed on JOSS website
- Discoverable via CrossRef

## Useful Links

- **JOSS Homepage**: https://joss.theoj.org/
- **JOSS Author Guide**: https://joss.readthedocs.io/en/latest/submitting.html
- **JOSS Review Criteria**: https://joss.readthedocs.io/en/latest/review_criteria.html
- **Example JOSS Papers**: https://joss.theoj.org/papers
- **JOSS on GitHub**: https://github.com/openjournals/joss
- **ORCID Registration**: https://orcid.org/register

## FAQ

**Q: Does JOSS cost money?**
A: No, JOSS is completely free for authors.

**Q: How long is the review process?**
A: Typically 1-3 months, but can be faster for well-prepared submissions.

**Q: Can I update my software after submission?**
A: Yes, you can update during review. Tag new versions appropriately.

**Q: What if reviewers request major changes?**
A: Rare, but possible. JOSS focuses on software utility and documentation, not novelty.

**Q: Can I withdraw my submission?**
A: Yes, you can withdraw at any time before acceptance.

**Q: What if my paper is rejected?**
A: Very rare. JOSS works with authors to improve submissions. Most papers are accepted after revisions.

## Support

If you have questions during the submission process:
- **JOSS Gitter**: https://gitter.im/openjournals/joss
- **JOSS Email**: joss@theoj.org
- **GitHub Issues**: https://github.com/openjournals/joss/issues

---

## Final Checklist Before Submitting

- [ ] ORCID updated in paper.md and CITATION.cff
- [ ] PyPI package published and installable
- [ ] GitHub release v0.1.0 created
- [ ] All tests passing
- [ ] All examples work
- [ ] Documentation complete
- [ ] Paper compiles without errors (test with Docker if possible)
- [ ] References formatted correctly in paper.bib
- [ ] README.md up to date
- [ ] LICENSE file present

Once all items are checked, you're ready to submit! 

Good luck with your JOSS submission!
