# CLSA_AI: CLSA AI-CVD analysis code (DeepECG-SSL, incident CVD)

Analysis code for "A lab-free electrocardiogram foundation model for cardiovascular risk triage in two Canadian cohorts"
(Bahani et al.; manuscript v3.2 for npj Digital Medicine, November 2026). It covers the reanalysis from the October 2026
incident-CVD relabel through the npj revision.

Inference with the released model does not live here. It is the `clsa_cvd` task of
[HeartWise-AI/DeepECG_Docker](https://github.com/HeartWise-AI/DeepECG_Docker) (merged in commit `8fc245ee`), and the
fine-tuned weights are on Hugging Face at `heartwise/DeepECG-SSL-CLSA-CVD` (revision `12be54ec`).

## No data in this repository

All inputs are read from the lab NAS and never committed. CLSA data are available from the CLSA under its data access
policy (application 2401002); MHI data cannot be shared.

| Input | Location |
|---|---|
| Raw CLSA files and dictionaries | `/media/data1/mbahani/CLSA/CLSA_ALL/CLSA_Data/CLSA/2401002_MontrealHeartInstitute_RAvram_{BL,FUP1,FUP2,Mortality}` |
| Labels, cohort, saved predictions | `/media/data1/ravram/CLSA_validation_files/manuscript/` (`incident_cvd_labels_v3_guideline_components.csv`, `data_v4/`, `results_v5/`, `mhi_anyecg/`, `v2/indeterminate/`) |
| Released model and parameters | `/media/data1/models/DeepECG-SSL/DeepECG_SSL_WCRv2_CLSA_CVD_v1/` |

Scripts that write outputs use `CLSA_WORKSPACE` (default `/volume/CLSA`) and write to
`$CLSA_WORKSPACE/outputs/revision/`. The MHI scripts also read `$CLSA_WORKSPACE/DeepECG_Docker/clsa_cvd/model_params.json`.

## Layout

| Folder | What it does | Manuscript |
|---|---|---|
| `pipeline/` | Code snapshot released with the model (October 7, 2026): baseline risk factors, cohort, guideline risk scores (`risk_scores.py`, `compute_scores_v4.py`), fine-tuning (`stage1b_wcr_cv.py`), test evaluation, guideline endpoints, MHI cohort, outcome rules, adjudication dossiers, MHI Biobank FRS, scoring of participants with indeterminate F2 status | Methods, Tables 1 to 4, Figures 2 to 4 |
| `revision/` | v2.8 to v3.0 analyses: ROC bands and agreement, MHI thresholds and calibration review, fairness, clinical impact, outcome validity A1 to A4, ECG diagnostic labels, pretraining-overlap audit | Supplementary analyses (v2.8 to v3.0) |
| `npj/` | npj revision, Steps 1 to 5b: ECG added to guideline scores, specific-CVD row, MHI recalibration and decision curve, MHI time-to-event and Fine-Gray (`npj_step4_finegray.R`), CLSA follow-up loss and discrete-time model | Results "Lipids add little to the ECG...", "Time-to-event analyses" |
| `docs/` | MHI outcome adjudication charter | Methods, external validation |

Run order: `pipeline/build_baseline_risk_factors.py`, `build_v4_dataset.py`, `compute_scores_v4.py`,
`stage1b_wcr_cv.py`, `final_test_evaluation_wcr.py`, `simple_models_primary.py`, `guideline_endpoints.py`
(`AIECG=wcr`), `guideline_endpoints_bootstrap.py`, then the MHI scripts (`mhi_anyecg_build.py`, `mhi_anyecg_infer.py`,
`mhi_anyecg_eval.py`), then `revision/` and `npj/` in numeric order. The npj steps were specified in writing before
they were run (internal analysis plan, section 8).

## Environment

Python 3.12 with numpy, pandas, scikit-learn, scipy, statsmodels, lifelines 0.30 and scikit-survival 0.28. R with the
`survival` package (Fine-Gray) and `preventr` 0.12.0 (PREVENT equations). Fine-tuning and ECG inference need
fairseq-signals (`/media/data1/ravram/CLSA_validation_files/manuscript/tools/venv_fss`).

## Tests and reproducibility

- `python pipeline/risk_scores.py` checks PCE 2013 and FRS 2008 against the published worked examples (Goff 2013;
  D'Agostino 2008).
- `cd revision && python -m pytest -q tests` runs the pretraining-overlap audit tests (13 tests).
- On November 19, 2026, `npj/` steps 1, 2, 3, 5 and 5b were rerun from this folder into an empty workspace and
  reproduced the stored outputs byte for byte. Step 4 (about 25 minutes) was not rerun.

## Known gaps

- The scripts that produced outcome labels v1 to v3 (`incident_cvd_labels_v1_F2primary.csv`,
  `..._v2_components.csv`, `..._v3_guideline_components.csv`) were not saved; only the draft builder
  (`pipeline/build_incident_cvd_labels_draft.py`) is here. The label files themselves are on the NAS.
- Paths are absolute to the lab server.

See `SKILL.md` for the agent-facing procedure to rerun or extend these analyses.

## Licence

Code: MIT (see `LICENSE`). The licence covers this code only; CLSA and MHI data remain under their own access terms, and model weights under the terms of their Hugging Face repository.
