# Changelog

## [0.2.2] - 2026-03-23

### Fixed
- Bumped numpy minimum version from >=1.24 to >=1.25 to ensure compatibility with scipy's use of numpy.exceptions (added in numpy 1.25)


## v0.2.0 (2026-03-22) [unreleased]
- feat: add Databricks benchmark notebook — SDID vs two-period DiD
- fix: correct license badge; add missing Issues and Documentation URLs
- fix: use plain string license field for universal setuptools compatibility
- fix: use importlib.metadata for __version__ (prevents drift from pyproject.toml)
- Add DRSC verification notebook (Databricks-verified 2026-03-21)

## v0.2.0 (2026-03-21)
- Add cross-links to related libraries in README
- Add DRSC vs SDID benchmark with Databricks-verified results
- Add DoublyRobustSCEstimator (Sant'Anna, Shaikh, Syrgkanis 2025)
- docs: replace pip install with uv add in README
- Add blog post link to README
- fix: soften EP25/2 methodology framing in evidence footer
- Add MIT license
- Add discussions link and star CTA
- Update Performance and Why bother sections with actual benchmark numbers
- Fix batch 11 audit issues: event study SEs, FCA claims, version sync
- Fix pandas 1.x compat and tighten regression test for P0-1
- Fix 3 P0 + 2 P1 bugs from deep code review
- Add consulting CTA to README
- Add benchmark: SDID vs naive before-after for rate change evaluation
- fix: bump scipy lower bound to >=1.10 for Python 3.12 wheel compatibility
- Polish flagship README: badges, benchmark table, problem statement
- Add worked example and Databricks notebook links to README
- Add worked example link to README
- Fix: pin scipy<1.11 and cvxpy<1.6.1 for Databricks serverless compat
- Add Related Libraries section to README
- Add SDID vs naive before-after benchmark notebook

