---
name: clsa-cvd-reanalysis
description: Rerun or extend the CLSA AI-CVD analyses (DeepECG-SSL lab-free incident-CVD model versus FRS, PREVENT, SCORE2 and PCE; MHI external validation). Use when asked to reproduce a manuscript number, add an analysis for a reviewer, or update the npj Digital Medicine revision of the CLSA paper.
---

# CLSA AI-CVD reanalysis

## Before you run anything

1. Read `README.md` in this folder and the internal analysis plan
   (`/media/data1/ravram/CLSA_validation_files/manuscript/REANALYSIS_PLAN.md`), especially section 8 (npj steps).
2. The outcome is incident CVD by Follow-up 2 (`incident_F2_primary`) in participants free of CVD and TIA at
   baseline. Never use the 2025 root-level code or any label built from baseline-prevalent cases.
3. Work on frozen predictions (`results_v5/`, `mhi_anyecg/predictions.parquet`, `v2/indeterminate/`). Do not retrain
   or reselect models for a revision analysis.

## Adding an analysis

1. Write its specification (population, endpoint, models, metric, decision rule) as a dated subsection of section 8 of
   the analysis plan before running it.
2. Reuse what exists: `revision/remaining_analyses.py` (`delong_ci`), `pipeline/final_test_evaluation.py`
   (`auc_ci`, `delong_test`), the score combinations in `pipeline/guideline_endpoints.py`, and the MHI time-to-event
   cohort in `npj/npj_step4_mhi_tte.py`.
3. Put the script in `npj/` (or a new revision folder), write outputs under `$CLSA_WORKSPACE/outputs/revision/`, and
   use paired bootstrap with 1,000 resamples and seed 2026 unless the plan says otherwise.
4. Report every result, including nulls. Aggregate outputs only; never write participant-level data into the repo.

## Checks before a commit

- `python pipeline/risk_scores.py` and `cd revision && python -m pytest -q tests` pass.
- Rerun any changed `npj/` step into an empty `CLSA_WORKSPACE` and compare with the stored outputs.
- No data, IDs, tokens or notebooks with outputs in the diff.

## Manuscript edits

Tracked-change manuscripts are built outside this repo, from the submitted Word file. Citations are reference-manager
content controls, so edits must not span them. Follow the lab skills Scientific Manuscript Editing and DOCX Revision
Operations (Notion, HeartWise AI Skills).
