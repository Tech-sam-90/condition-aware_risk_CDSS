# Data Processing

This folder contains scripts and pipelines for extracting, cleaning, and aligning raw MIMIC-IV data.

## Scripts

- `prepare_condition_cohort.py`: builds condition cohort and stay-level condition table.

## Notebooks

- `data_processing.ipynb`: exploratory/interactive processing notebook.
- `icd_code_validation.ipynb`: merges `diagnoses_icd` with `d_icd_diagnoses`, prints ICD code lists for sepsis/heart failure/CKD/diabetes, and exports review CSVs.
- `icd_code_filter_non_neonatal.ipynb`: filters neonatal/newborn-related ICD titles from the reviewed list and exports a non-neonatal codebook.

## Run

```bash
python data_processing/prepare_condition_cohort.py
```

Then run notebooks (recommended order):

1. `icd_code_validation.ipynb`
2. `icd_code_filter_non_neonatal.ipynb`

## Outputs

Main processing outputs (created in `data/processed/`):

- `condition_cohort_table.csv`
- `condition_model_table_v1.csv`

ICD review outputs (created in `data/processed/icd_review/`):

- `icd_codes_for_4_conditions_full.csv`
- `icd_codes_for_4_conditions_summary.csv`
- `icd_codes_title_only_candidates.csv`
- `icd_codes_for_4_conditions_non_neonatal.csv`
- `icd_codes_for_4_conditions_non_neonatal_summary.csv`
