# Data

`openpowerlifting.csv`: 4,050,248 rows × 42 columns, comma-separated, 828 MB (export of 2026-10-17).

**Source:** [OpenPowerlifting](https://www.openpowerlifting.org/) data service, https://openpowerlifting.gitlab.io/opl-csv/.
The data is public domain (see `LICENSE.txt` in the download). Results are typed in from federation result sheets by
OpenPowerlifting volunteers; each row is one lifter at one meet, with every attempt, best lifts, total, placing and points.

**The CSV is too big for the repo**, so it is not committed. `src/clean.py` looks for it in this order:

1. `data/openpowerlifting.csv` (this folder)
2. `../Data/openpowerlifting-2026-10-17/openpowerlifting.csv` (the shared `Data` folder next to the project)

On first use, `load_attempts()` builds `data/attempts_ipf.pkl` (6,486,914 attempts, about 460 MB) so later notebooks load
in seconds. The cache is ignored by git and can be rebuilt any time with `load_attempts(rebuild=True)`.

See `notebooks/01_data_overview.ipynb` for the full data dictionary.
