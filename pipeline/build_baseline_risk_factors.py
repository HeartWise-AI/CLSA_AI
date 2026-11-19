"""Extract baseline (BL, CoPv7) risk-factor variables for risk scores from raw CLSA.
Output: baseline_risk_factors.parquet (one row per entity_id)."""
import os, re, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BL = "/media/data1/mbahani/CLSA/CLSA_ALL/CLSA_Data/CLSA/2401002_MontrealHeartInstitute_RAvram_BL/2401002_MontrealHeartInstitute_RAvram_Baseline_CoPv7.csv"
BASE = ["entity_id", "AGE_NMBR_COM", "SEX_ASK_COM", "BP_SYSTOLIC_AVG_COM", "HBP_MEDCUR_COM", "HBP_MED_COM", "CCC_HBP_COM",
        "BLD_CHOL_COM", "BLD_HDL_COM", "BLD_nonHDL_COM", "BLD_LDL_COM", "BLD_CREAT_COM", "BLD_HBA1c_COM",
        "DIA_DIAB_COM", "DIA_MEDCUR_COM", "DIA_TYPE_COM", "SMK_DSTY_COM", "HWT_DBMI_COM", "WGHTS_PROV_COM"]

hdr = pd.read_csv(BL, nrows=0).columns
atc = [c for c in hdr if re.fullmatch(r"MEDI_ID_ATC_\d+_COM", c)]
cult = [c for c in hdr if re.fullmatch(r"SDC_CULT_[A-Z]+_COM", c)]
d = pd.read_csv(BL, usecols=BASE + atc + cult, low_memory=False).set_index("entity_id")


def num(s, lo=None, hi=None):
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s >= 0)  # CLSA missing codes are negative (-8, -88888, ...)
    if lo is not None: s = s.where(s >= lo)
    if hi is not None: s = s.where(s <= hi)
    return s


codes = d[atc].astype(str).apply(lambda col: col.str.upper().str.strip())
has = lambda pat: codes.apply(lambda col: col.str.match(pat, na=False)).any(axis=1)

out = pd.DataFrame(index=d.index)
out["age"] = num(d.AGE_NMBR_COM, 18, 110)
out["female"] = (d.SEX_ASK_COM == "F").astype(int)
out["sbp"] = num(d.BP_SYSTOLIC_AVG_COM, 60, 260)
out["tc_mmol"] = num(d.BLD_CHOL_COM, 1, 20)
out["hdl_mmol"] = num(d.BLD_HDL_COM, 0.2, 5)
out["nonhdl_mmol"] = num(d.BLD_nonHDL_COM, 0, 20)
out["ldl_mmol"] = num(d.BLD_LDL_COM, 0, 15)
out["creat_umol"] = num(d.BLD_CREAT_COM, 10, 2000)
out["hba1c_pct"] = num(d.BLD_HBA1c_COM, 3, 20)
out["bmi"] = num(d.HWT_DBMI_COM, 10, 80)
smk = pd.to_numeric(d.SMK_DSTY_COM, errors="coerce")
out["smoker_current"] = smk.isin([1, 2, 3]).astype(float).where(smk.isin([1, 2, 3, 4, 5, 6]))
out["statin"] = has(r"C10AA|C10B").astype(int)
out["lipid_lowering_any"] = has(r"C10").astype(int)
out["antihtn_atc"] = has(r"C02|C03|C07|C08|C09").astype(int)
out["bp_treated"] = ((pd.to_numeric(d.HBP_MEDCUR_COM, errors="coerce") == 1) |
                     (pd.to_numeric(d.HBP_MED_COM, errors="coerce") == 1)).astype(int)
out["glucose_lowering"] = (has(r"A10") | (pd.to_numeric(d.DIA_MEDCUR_COM, errors="coerce") == 1)).astype(int)
dia_self = pd.to_numeric(d.DIA_DIAB_COM, errors="coerce")
dia_type = pd.to_numeric(d.DIA_TYPE_COM, errors="coerce")
# diabetes: physician-diagnosed (excluding "borderline/neither" type answers), glucose-lowering drug, or HbA1c >= 6.5%
out["diabetes"] = (((dia_self == 1) & ~dia_type.isin([3])) | (out.glucose_lowering == 1) | (out.hba1c_pct >= 6.5)).astype(int)
out["hypertension_dx"] = (pd.to_numeric(d.CCC_HBP_COM, errors="coerce") == 1).astype(int)
for r in ["WH", "BL", "ZH", "SA", "FP", "LA", "SE", "AR", "WA", "JA", "KO", "OT"]:
    c = f"SDC_CULT_{r}_COM"
    if c in d: out[f"cult_{r}"] = (pd.to_numeric(d[c], errors="coerce") == 1).astype(int)
out["race_pce"] = np.where(out.get("cult_BL", 0) == 1, "AA", "WH")  # PCE: Black -> AA equations; all others -> White
out["province"] = d.WGHTS_PROV_COM


def egfr_ckdepi2021(scr_umol, age, female):
    scr = scr_umol / 88.4
    k = np.where(female == 1, 0.7, 0.9); a = np.where(female == 1, -0.241, -0.302)
    return 142 * np.minimum(scr / k, 1) ** a * np.maximum(scr / k, 1) ** -1.200 * 0.9938 ** age * np.where(female == 1, 1.012, 1)


out["egfr"] = egfr_ckdepi2021(out.creat_umol, out.age, out.female)
out.to_parquet(os.path.join(HERE, "baseline_risk_factors.parquet"))
print(out.shape)
print(out.describe().T[["count", "mean", "50%", "min", "max"]].round(2).to_string())
