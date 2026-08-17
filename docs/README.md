# Documentation

Written deliverables for the Cryptocurrency Market Analysis System.

| File | What it is |
|---|---|
| `Cryptocurrency_Market_Analysis_System_SRS.docx` | Software Requirements Specification — functional and non-functional requirements, data sources, constraints, and an object-oriented analysis/design appendix (use-case, class, and sequence diagrams) |
| `Cryptocurrency_Market_Analysis_System_Report.docx` | Project report — methodology, implementation, results, limitations |

Elsewhere in the repository:

| Location | What it is |
|---|---|
| `../README.md` | Setup, quick start, project structure |
| `../CHANGES.md` | Every correctness fix made to the codebase, with reasoning |
| `../notebooks/exploration.ipynb` | Reproducible walkthrough using the same analysis modules as the dashboard |

## Known gap between the report and the code

**Section 7.3 of the report — the market-cycle rule table — no longer matches
the implementation.** Two rules were tightened after the report was written:

- Distribution now additionally requires price to be within 10% of the peak.
  Previously any uptrend with soft momentum was labelled Distribution regardless
  of where price sat, which is not what the phase describes.
- Accumulation now requires a drawdown past 25% to already be in place. A
  shallow, early decline previously read as a basing bottom.

The current table is reproduced in the dashboard under *Market Cycles → How
phases are identified*, and in `CHANGES.md` section 8. Copy it into the report
before submitting — a mismatch between the two is the kind of thing an examiner
notices.
