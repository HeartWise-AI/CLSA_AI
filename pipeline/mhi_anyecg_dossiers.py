"""Blinded adjudication sample for the any-ECG cohort: 400 rule-positive + 200 rule-negative patients from the primary
analysis set (random, seed 2026). De-identified dossiers (relative days only) -> mhi_anyecg/dossiers/; key -> sample_key.parquet."""
import os, glob, json, numpy as np, pandas as pd
A = "/media/data1/gjabbour/Admin_data/"; O = "mhi_anyecg"; os.makedirs(O + "/dossiers", exist_ok=True)
mrn = lambda s: pd.to_numeric(s.astype(str).str.replace(r"\D", "", regex=True), errors="coerce")
c = pd.read_parquet(f"{O}/cohort_labels.parquet")
pa = c[(c.prev == "no") & ~c.unk_death & ((c.y == 1) | c.fu_ecg_6y)]
s = pd.concat([pa[pa.y == 1].sample(400, random_state=2026), pa[pa.y == 0].sample(200, random_state=2026)]).sample(frac=1, random_state=1)
s["sid"] = [f"AE{i:04d}" for i in range(len(s))]; s.to_parquet(f"{O}/sample_key.parquet"); ids = set(s.MRN); t0 = s.set_index("MRN").t0
dx = []
for f in sorted(glob.glob(A + "CSV/1_Diagnostics_MedEcho_*.csv")):
    x = pd.read_csv(f, sep=";", dtype=str); x["MRN"] = mrn(x.NDOS); dx.append(x[x.MRN.isin(ids)].drop(columns="NDOS"))
dx = pd.concat(dx).drop_duplicates(["Episode", "ID Dx"])
me = pd.read_parquet("/media/data1/datasets/Labels_v5_MedEcho/cache/medecho_dx_2015_2022.parquet") if os.path.exists("/media/data1/datasets/Labels_v5_MedEcho/cache/medecho_dx_2015_2022.parquet") else pd.read_parquet("/media/data1/datasets/VentricAI/Labels_v5_MedEcho/cache/medecho_dx_2015_2022.parquet")
dx = dx.merge(me[["Episode", "ID Dx", "DESCDIAGNOSTIC", "DESCCARACTERISTIQUE"]], on=["Episode", "ID Dx"], how="left")
dx["MRN"] = dx.MRN.astype(np.int64); dx["d"] = (pd.to_datetime(dx.StartDateTime, format="mixed").dt.normalize() - dx.MRN.map(t0)).dt.days
dx["de"] = (pd.to_datetime(dx.EndDateTime, format="mixed").dt.normalize() - dx.MRN.map(t0)).dt.days; dx["seq"] = pd.to_numeric(dx.SEQUENCE, errors="coerce")
bloc = pd.read_csv(A + "CSV/5_Bloc_operatoire_2002-2022.csv", sep=";", dtype=str, encoding="latin-1"); bloc["MRN"] = mrn(bloc.NDOS); bloc = bloc[bloc.MRN.isin(ids)].copy()
bloc["d"] = (pd.to_datetime(bloc["Date de la chirurgie"], format="mixed") - bloc.MRN.astype(np.int64).map(t0)).dt.days
hemo = pd.read_parquet("/tmp/claude-0/-volume-CLSA/78bfb225-0661-4cca-8d0d-9b5dd910b130/scratchpad/hemo.parquet"); hemo["MRN"] = mrn(hemo.NDOS); hemo = hemo[hemo.MRN.isin(ids)].copy()
hemo["d"] = (pd.to_datetime(hemo["Date de l'intervention"], format="mixed").dt.normalize() - hemo.MRN.astype(np.int64).map(t0)).dt.days
pt = pd.read_parquet("/media/data1/gjabbour/df/df_mhi_ptindex.parquet").drop_duplicates("id").set_index("id")
for _, r in s.iterrows():
    m = r.MRN; g = dx[dx.MRN == m].sort_values(["d", "Episode", "seq"])
    hist = sorted(set(g[g.d < -365].CDDIAGNOSTIC.dropna()))
    stays = [dict(admit_day=int(q.d.iloc[0]), discharge_day=int(q.de.iloc[0]),
                  dx=[dict(seq=None if pd.isna(z.seq) else int(z.seq), code=z.CDDIAGNOSTIC, desc=z.DESCDIAGNOSTIC if isinstance(z.DESCDIAGNOSTIC, str) else None,
                           flag=z.DESCCARACTERISTIQUE if isinstance(z.DESCCARACTERISTIQUE, str) else None) for z in q.itertuples()])
             for _, q in g[(g.d >= -365) & (g.d <= 2190)].groupby("Episode", sort=False)]
    dd = pt.death_date.get(m, pd.NaT)
    dos = dict(study_id=r.sid, age=int(r.age), sex="F" if r.female else "M", day0="index ECG (MHI MUSE, 2006-2016)", horizon_days=2190,
               data_coverage="MedEcho MHI stays (codes all years; French descriptions + Complication flag only for stays 2015+), OR surgeries 2002-2022, cath-lab procedures 2011-2022, deaths to 2023-01. Events at other hospitals invisible.",
               prior_history_codes_older_than_1y=hist, mhi_stays_from_minus365=stays,
               or_surgeries=[dict(day=int(z.d), desc=z.Description) for z in bloc[bloc.MRN == m].sort_values("d").itertuples()],
               cathlab_procedures=[dict(day=int(z.d), desc=z.Description) for z in hemo[hemo.MRN == m].sort_values("d").itertuples() if pd.notna(z.d)],
               death_day=None if pd.isna(dd) else int((dd - r.t0).days), last_followup_ecg_day=int(r.last_ecg_day))
    json.dump(dos, open(f"{O}/dossiers/{r.sid}.json", "w"), ensure_ascii=False, default=str)
print("dossiers", len(s))
