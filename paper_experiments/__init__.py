"""
Paper experiments: extension of the Hybrid FD CDF thesis benchmark.

Code lives here; results go to results/paper_experiments/<part>/ (never into results/exp*).
Run modules from the repo root as `poetry run python -m paper_experiments.<module>`.

Shared contract:
  provenance.py  git commit / work-tree hash / file hashes / package versions -> manifest.json
  schema.py      unified row identifiers, stages (STATIC / SHOCK / FT / RB), units,
                 and the drift_steps.csv -> long-format converter every exporter uses
"""
