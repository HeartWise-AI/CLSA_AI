# MHI any-ECG cohort — blinded outcome adjudication charter (2026-10-05)
You are BLINDED to the algorithmic label. Day 0 = an MHI ECG (2006-2016). Window = day 8 to day 2190 (6 years).
Dossier: MHI stays from day -365 with ICD-10-CA codes (seq 1 = principal; French descriptions and the "Complication"
flag exist only for stays from 2015 on; older stays have codes only), history codes > 1 y before day 0 (ICD-9 before
2006), OR surgeries (all years), cath-lab procedures (2011+), death day, last follow-up ECG day.
Only MHI admissions are visible.

Questions:
1. prevalent_cvd (yes/no/uncertain) at day 0: IHD (I20-I25, 410-414), HF (I50, 428), stroke/TIA (I60-I64, I69, G45,
   430-438), PAD (I70.2, I73.9), prior PCI/CABG (Z95.1, Z95.5, procedures) recorded in any stay admitted on/before day
   7 (incl. the stay containing day 0 and presenting events), or as chronic codes in a stay admitted day 8-30, or CABG/PCI
   on/before day 30. UNCERTAIN if prior disease is strongly implied but undated (I25.2, Z95.1, Z95.5, I69, Z86.7x first
   coded later without a matching procedure after day 0).
2. primary_event (1/0/uncertain), day 8-2190: first of acute MI (I21/I22 any position, not I25.2), stroke (I60-I64 any
   position; not TIA, not retinal occlusion), HF hospitalisation (HF is the PRINCIPAL reason for admission; post-
   operative/post-procedural HF and I50 as a secondary comorbidity do NOT count), CV death (death inside an MHI stay).
   Death outside MHI = unknown cause, not an event. An I21 within 28 d of an MI stay is the same MI.
3. primary_type, primary_day.
4. death_category: cv_in_mhi / unknown_outside_mhi / none_in_window.
Judge clinically; do not invent; "uncertain" with a reason when evidence conflicts.
Output JSONL keys exactly: study_id, prevalent_cvd, prevalent_reason, primary_event, primary_type, primary_day,
death_category, confidence (high/medium/low), rationale (<= 30 words)
