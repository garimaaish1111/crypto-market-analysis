# Documentation

Written deliverables for the Cryptocurrency Market Analysis System.

| File | What it is |
|---|---|
| `Cryptocurrency_Market_Analysis_System_SRS.docx` | Software Requirements Specification — scope, functional and non-functional requirements, data sources, constraints, and an object-oriented analysis and design appendix |
| `Cryptocurrency_Market_Analysis_System_Report.docx` | Project report — methodology, implementation, validation, results, and limitations |
| `Crypto_Market_Analysis_Final_Evaluation.pptx` | Final evaluation deck — problem, architecture, methodology, dashboard walkthrough, and findings |

Elsewhere in the repository:

| Location | What it is |
|---|---|
| `../README.md` | Setup, how to run, project structure |
| `../CHANGES.md` | Engineering log — defects found, and the reasoning behind each fix |
| `../notebooks/exploration.ipynb` | Reproducible walkthrough calling the same analysis functions as the dashboard |
| `../tests/` | 209 tests covering the analysis layer at 92–97% |

## Consistency

The written deliverables, the code, and the dashboard's own explanatory panels
describe the same system. Where a document states a rule, a threshold, or a
metric, that value is read from `config.py` or reproduced from the implementation
it documents.

Two conventions are worth stating once, because they affect how every figure in
these documents should be read:

- **Denomination is Indian rupees.** Crypto prices are fetched from CoinGecko
  directly in INR. Dollar-quoted traditional assets are converted at the daily
  USD/INR rate; the US Dollar Index and the 10-year yield are not converted,
  because an index level and a rate in percentage points are not prices.
- **Figures are a dated snapshot.** The system ships with a populated cache of
  real API responses. The Datasets tab reports the exact date range covered by
  every feed, so any number in these documents is attributable to a moment in
  time rather than presented as perpetually current.
