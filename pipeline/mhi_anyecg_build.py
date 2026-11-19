"""MHI external validation, ANY-ECG cohort with 6-year MedEcho outcomes (2026-10-05, per RA: "use any ECG, use MedEcho",
6-y window = CLSA F2 horizon, non-events need ECG follow-up, no post-op HF).

Pre-specified (before any model output on this cohort):
  Index ECG: each patient's FIRST MUSE ECG (any reason/location) dated 2006-04-01..2016-03-30 (MedEcho ICD-10 era; the
    6-y window must end before MedEcho stops on 2022-03-30). Age 18-100 at the ECG. Waveform .npy must exist.
  Prevalent CVD (excluded): MedEcho codes (ICD-9/10, 1995+) for IHD / HF / stroke-TIA / PAD / prior PCI-CABG in any stay
    admitted <= day 7 (incl. the stay containing the ECG and presenting events); chronic codes of that list in stays
    admitted day 8-30 (ECG = work-up), except acute qualifying events (I21/I22, I60-I64, principal I50); CABG (Bloc
    'PONTAGE') or PCI (Hemodynamie 'ANGIOPLASTIE'/'TUTEUR', 2011+) <= day 30.
  Prevalent-uncertain (excluded from primary, kept in sensitivity): undated history codes (old MI I25.2, CABG status Z95.1,
    PCI status Z95.5, sequelae of stroke I69, personal history Z86.7x) first coded after day 30 and before any primary
    event, without a matching CABG/PCI procedure recorded after day 0 (Z95.1/Z95.5 only).
  Primary outcome, day 8 to day 2190 (6 y): first of MI (I21/I22 any position, MHI stay admitted in window), stroke
    (I60-I64 any position), HF hospitalisation (I50, or I11.0/I13.0/I13.2 with I50, as PRINCIPAL diagnosis only -> post-
    operative HF never counts), CV death (death inside an MHI stay).
  Death outside MHI before any event = unknown cause -> excluded (sensitivity: non-event).
  Non-events must have a later MUSE ECG at >= day 2190 (documented follow-up); events kept regardless.
  Broad (secondary): primary + CABG/PCI (excluding day 0-90 = index work-up) + unstable angina (I20.0 principal).
Outputs (confidential, NAS): mhi_anyecg/cohort_labels.parquet, summary.json."""
import os, glob, json, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, os.environ.get("MHI_OUT", "mhi_anyecg")); os.makedirs(OUT, exist_ok=True)
A = "/media/data1/gjabbour/Admin_data/"; W0, W1, H = pd.Timestamp("2006-04-01"), pd.Timestamp(os.environ.get("MHI_W1", "2016-03-30")), 2190  # MHI_W1 > 2016-03 only for time-to-event (censor at 2022-03-30)
mrn = lambda s: pd.to_numeric(s.astype(str).str.replace(r"\D", "", regex=True), errors="coerce")
P = lambda c, L: c.str.startswith(tuple(L))

# ---- ECGs
e = pd.read_parquet("/media/data1/muse_ge/ECG_ad20241231_cat_labels_v1.3.parquet",
                    columns=["new_PatientID", "RestingECG_TestDemographics_AcquisitionDate", "RestingECG_TestDemographics_AcquisitionTime", "npy_path",
                             "RestingECG_PatientDemographics_PatientAge", "RestingECG_PatientDemographics_Gender"])
e["MRN"] = pd.to_numeric(e.new_PatientID, errors="coerce"); e["date"] = pd.to_datetime(e.RestingECG_TestDemographics_AcquisitionDate, errors="coerce", format="mixed")
e = e.dropna(subset=["MRN", "date"]); e["MRN"] = e.MRN.astype(np.int64)
last_ecg = e.groupby("MRN").date.max()
if os.environ.get("MHI_INDEX"):  # index override (e.g. MHI Biobank: ECG nearest enrolment); parquet with MRN, enrol
    ix = pd.read_parquet(os.environ["MHI_INDEX"])[["MRN", "enrol", "Age baseline"]].dropna(subset=["MRN", "enrol"]); ix["MRN"] = ix.MRN.astype(np.int64)
    c = e.merge(ix, on="MRN"); c["gap"] = (c.date - c.enrol).dt.days.abs()
    c = c[(c.gap <= int(os.environ.get("MHI_GAP", 180))) & (c.date >= W0) & (c.date <= W1)].sort_values(["MRN", "gap"]).groupby("MRN").first().reset_index()
else:
    c = e[(e.date >= W0) & (e.date <= W1)].sort_values(["MRN", "date", "RestingECG_TestDemographics_AcquisitionTime"]).groupby("MRN").first().reset_index()
_pt = pd.read_parquet("/media/data1/gjabbour/df/df_mhi_ptindex.parquet").drop_duplicates("id").set_index("id")  # MUSE age is 82% missing -> MHI patient index
c["age"] = c.date.dt.year - c.MRN.map(_pt.birth_year)
if "Age baseline" in c: c["age"] = c.age.fillna(c["Age baseline"] + (c.date - c.enrol).dt.days / 365.25)
c["age"] = c.age.fillna(pd.to_numeric(c.RestingECG_PatientDemographics_PatientAge, errors="coerce"))
sx = c.MRN.map(_pt.sex).str.upper().fillna(c.RestingECG_PatientDemographics_Gender.str.upper()); c = c[sx.isin(["FEMALE", "MALE"])].copy(); c["female"] = (sx[c.index] == "FEMALE").astype(int)
n0 = len(c); c = c[c.age.between(18, 100)].copy()
c["t0"] = c.date.dt.normalize(); c["last_ecg_day"] = (c.MRN.map(last_ecg).dt.normalize() - c.t0).dt.days
t0 = c.set_index("MRN").t0; ids = set(c.MRN)
print("index ECGs:", n0, "| age 18-100:", len(c))

# ---- MedEcho (CSV 1995-2022: codes) ; deaths ; procedures
dx = []
for f in sorted(glob.glob(A + "CSV/1_Diagnostics_MedEcho_*.csv")):
    x = pd.read_csv(f, sep=";", dtype=str, usecols=["Episode", "NDOS", "StartDateTime", "EndDateTime", "ID Dx", "SEQUENCE", "CDDIAGNOSTIC"])
    x["MRN"] = mrn(x.NDOS); dx.append(x[x.MRN.isin(ids)].drop(columns="NDOS"))
dx = pd.concat(dx).drop_duplicates(["Episode", "ID Dx"])
dx["MRN"] = dx.MRN.astype(np.int64); dx["start"] = pd.to_datetime(dx.StartDateTime, format="mixed").dt.normalize(); dx["end"] = pd.to_datetime(dx.EndDateTime, format="mixed").dt.normalize()
dx["code"] = dx.CDDIAGNOSTIC.fillna("").str.upper().str.replace(".", "", regex=False); dx["seq"] = pd.to_numeric(dx.SEQUENCE, errors="coerce")
dx["d"] = (dx.start - dx.MRN.map(t0)).dt.days; dx["de"] = (dx.end - dx.MRN.map(t0)).dt.days
pt = pd.read_parquet("/media/data1/gjabbour/df/df_mhi_ptindex.parquet"); death = pt.set_index("id").death_date
bloc = pd.read_csv(A + "CSV/5_Bloc_operatoire_2002-2022.csv", sep=";", dtype=str, encoding="latin-1"); bloc["MRN"] = mrn(bloc.NDOS)
bloc = bloc[bloc.MRN.isin(ids) & bloc.Description.str.upper().str.contains("PONTAGE AORTO|PONTAGE MAMMARO", na=False)].copy(); bloc["MRN"] = bloc.MRN.astype(np.int64)
bloc["d"] = (pd.to_datetime(bloc["Date de la chirurgie"], format="mixed") - bloc.MRN.map(t0)).dt.days
hemo = pd.read_excel(A + "6_Hémodynamie 2011-2022.xlsx", dtype=str); hemo["MRN"] = mrn(hemo.NDOS)
hemo = hemo[hemo.MRN.isin(ids) & hemo.Description.str.upper().str.contains("ANGIOPLASTIE|TUTEUR", na=False)].copy(); hemo["MRN"] = hemo.MRN.astype(np.int64)
hemo["d"] = (pd.to_datetime(hemo["Date de l'intervention"], format="mixed").dt.normalize() - hemo.MRN.map(t0)).dt.days
rev = pd.concat([bloc[["MRN", "d"]].assign(k="CABG"), hemo[["MRN", "d"]].assign(k="PCI")])

PREV10 = ["I20", "I21", "I22", "I23", "I24", "I25", "I50", "I60", "I61", "I62", "I63", "I64", "I69", "G45", "I702", "I739", "Z951", "Z955"]
PREV9 = ["410", "411", "412", "413", "414", "428", "430", "431", "432", "433", "434", "435", "436", "438", "4402", "4439", "V4581", "V4582"]
PREV = PREV10 + PREV9
MI, STROKE = ["I21", "I22"], ["I60", "I61", "I62", "I63", "I64"]
# HF principal (I50 principal, or I11.0/I13.0/I13.2 principal with an I50 in the same episode)
ep_i50 = set(dx[P(dx.code, ["I50"])].Episode)
dx["hf_princ"] = (dx.seq == 1) & (P(dx.code, ["I50"]) | (P(dx.code, ["I110", "I130", "I132"]) & dx.Episode.isin(ep_i50)))
dx["acute_q"] = P(dx.code, MI + STROKE) | dx.hf_princ

# ---- prevalence
p1 = set(dx[(dx.d <= 7) & P(dx.code, PREV)].MRN)
p2 = set(dx[(dx.d > 7) & (dx.d <= 30) & P(dx.code, PREV) & ~dx.acute_q].MRN)
p3 = set(rev[rev.d <= 30].MRN)
prev = p1 | p2 | p3

# ---- events day 8..2190
w = dx[(dx.d >= 8) & (dx.d <= H)]
ev = pd.concat([w[P(w.code, MI)].assign(t="MI"), w[P(w.code, STROKE)].assign(t="stroke"), w[w.hf_princ].assign(t="HF")])[["MRN", "d", "t"]]
stays = dx.drop_duplicates("Episode")[["MRN", "start", "end"]]
c["death"] = c.MRN.map(death); c["dday"] = (c.death - c.t0).dt.days
dw = c[(c.dday >= 1) & (c.dday <= H)][["MRN", "death", "dday"]].merge(stays, on="MRN", how="left")  # fix (adjudication): deaths day 1-7 = presenting -> prevalent
in_mhi = dw[(dw.start <= dw.death) & (dw.end >= dw.death - pd.Timedelta(days=1))].MRN.unique()
early_death = set(c[c.MRN.isin(in_mhi) & (c.dday < 8)].MRN)
cvd = c[c.MRN.isin(in_mhi) & (c.dday >= 8)][["MRN", "dday"]].rename(columns={"dday": "d"}).assign(t="CV death")
ev = pd.concat([ev, cvd]).sort_values(["MRN", "d"])
first = ev.groupby("MRN").first()
c["y"] = c.MRN.isin(first.index).astype(int); c["event_day"] = c.MRN.map(first.d); c["event_type"] = c.MRN.map(first.t)
# broad: + revasc (day 91-2190) + UA principal
ua = w[(w.seq == 1) & P(w.code, ["I200"])][["MRN", "d"]].assign(t="UA")
bev = pd.concat([ev, rev[(rev.d > 90) & (rev.d <= H)].rename(columns={"k": "t"}), ua]).sort_values(["MRN", "d"]).groupby("MRN").first()
c["y_broad"] = c.MRN.isin(bev.index).astype(int)
# uncertain prevalence: undated history codes first seen after day 30 and before the first event
UND = ["I252", "Z951", "Z955", "I69", "Z867"]
u = dx[(dx.d > 30) & P(dx.code, UND)].copy(); u["fe"] = u.MRN.map(first.d)
u = u[u.fe.isna() | (u.d < u.fe)]
revd = rev[rev.d > 0].groupby(["MRN", "k"]).d.min().unstack()
def explained(r):
    if r.code.startswith("Z951"): return r.MRN in revd.index and pd.notna(revd.loc[r.MRN].get("CABG")) and revd.loc[r.MRN, "CABG"] <= r.d
    if r.code.startswith("Z955"): return r.MRN in revd.index and pd.notna(revd.loc[r.MRN].get("PCI")) and revd.loc[r.MRN, "PCI"] <= r.d
    return False
u = u[~u.apply(explained, axis=1)] if len(u) else u
prev = prev | early_death
ua_any = set(dx[(dx.d > 30) & P(dx.code, UND)].MRN) - set(u.MRN)  # adjudicator-style: undated codes at ANY time (incl. at/after the event stay)
c["prev"] = np.where(c.MRN.isin(prev), "yes", np.where(c.MRN.isin(set(u.MRN)), "unc", "no"))
c["unc_any"] = c.MRN.isin(ua_any) | c.MRN.isin(set(u.MRN))
c["unk_death"] = (c.dday >= 8) & (c.dday <= H) & ~c.MRN.isin(in_mhi) & (c.event_day.isna() | (c.event_day > c.dday))
c["fu_ecg_6y"] = c.last_ecg_day >= H
c["npy_ok"] = c.npy_path.map(lambda p: isinstance(p, str) and os.path.exists(p))
c[["MRN", "date", "t0", "npy_path", "age", "female", "prev", "y", "event_day", "event_type", "y_broad", "unk_death", "fu_ecg_6y", "last_ecg_day", "npy_ok", "unc_any"]].to_parquet(os.path.join(OUT, "cohort_labels.parquet"))
inc = c[c.prev == "no"]; prim = inc[~inc.unk_death & ((inc.y == 1) | inc.fu_ecg_6y) & inc.npy_ok]
s = dict(index_ecgs_2006_2016=n0, age_18_100=len(c), prevalent=c.prev.value_counts().to_dict(), incident=len(inc),
         unknown_death_excluded=int(inc.unk_death.sum()), nonevents_without_6y_ecg=int(((inc.y == 0) & ~inc.fu_ecg_6y & ~inc.unk_death).sum()),
         npy_missing=int((~c.npy_ok).sum()), primary_n=len(prim), primary_events=int(prim.y.sum()), event_types=prim[prim.y == 1].event_type.value_counts().to_dict(),
         broad_events=int(prim.y_broad.sum()), age_mean=round(prim.age.mean(), 1), female=round(prim.female.mean(), 3),
         index_year=prim.t0.dt.year.value_counts().sort_index().to_dict())
json.dump(s, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str); print(json.dumps(s, indent=1, default=str))
