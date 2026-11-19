import pandas as pd, numpy as np
S="/tmp/claude-0/-volume-CLSA/78bfb225-0661-4cca-8d0d-9b5dd910b130/scratchpad/"
b=pd.read_parquet(S+"raw_BL.parquet").set_index("entity_id"); f=pd.read_parquet(S+"raw_F1.parquet").set_index("entity_id"); g=pd.read_parquet(S+"raw_F2.parquet").set_index("entity_id")
COMP=["HEART","AMI","ANGI","CVA","CAB"]           # outcome components (manuscript: HF/CAD/stroke)
EXCL=COMP+["TIA"]                                  # baseline exclusion also removes prior TIA
def col(d,c,suf): return d[("IHD_CAB_" if c=="CAB" else f"CCC_{c}_")+suf]
out={}
# ---- baseline
blyes=pd.concat({c:col(b,c,"COM")==1 for c in EXCL},axis=1); blno=pd.concat({c:col(b,c,"COM")==2 for c in EXCL},axis=1)
n0=len(b); prev=blyes.any(axis=1); unk=~prev & ~blno.all(axis=1)
free=b.index[blno.all(axis=1)]
print(f"BL participants {n0}; prevalent CVD/TIA at BL {prev.sum()} ({prev.mean():.1%}); unknown/missing BL answers {unk.sum()}; CVD-free at BL {len(free)}")
print(" BL prevalent by component:",{c:int(blyes[c].sum()) for c in EXCL})
# ---- F1
ff=f.reindex(free)
nof1=ff[f"CCC_HEART_COF1"].isna() | (ff["CCC_HEART_COF1"]==-88880)  # no F1 record / no DCS visit
y1=pd.concat({c:col(ff,c,"COF1")==1 for c in COMP},axis=1); n1=pd.concat({c:col(ff,c,"COF1")==2 for c in COMP},axis=1)
case1=y1.any(axis=1); ctrl1=n1.all(axis=1)&~case1; und1=~case1&~ctrl1&~nof1
print(f"\nOf CVD-free at BL: no F1 DCS visit/record {nof1.sum()}; F1 CASES {case1.sum()}; F1 CONTROLS {ctrl1.sum()}; indeterminate {und1.sum()}")
print(f" cumulative incidence by F1 {case1.sum()/(case1.sum()+ctrl1.sum()):.2%}")
print(" F1 case components:",{c:int(y1[c].sum()) for c in COMP})
print(" cases with HEART only (no AMI/ANGI/CVA/CAB):",int((y1.HEART & ~y1[["AMI","ANGI","CVA","CAB"]].any(axis=1)).sum()))
hard=(y1.AMI|y1.CVA); print(" hard ASCVD (MI or stroke) cases:",int(hard.sum()))
tia=(col(ff,"TIA","COF1")==1); print(" TIA-only new events (not counted):",int((tia&~case1).sum()))
# recall check: age at dx earlier than BL age
age_bl=b.AGE_NMBR_COM.reindex(free)
for c,a in [("AMI","IHD_AMIAGE_NB_COF1"),("ANGI","IHD_ANGIAGE_NB_COF1"),("CVA","STR_CVAAGE_NB_COF1"),("HEART","CCC_HEARTAGE_NB_COF1")]:
    ag=ff[a]; m=y1[c]&(ag>0)&(ag<age_bl-1)
    print(f" {c}: F1 cases reporting dx age < BL age-1: {int(m.sum())} / {int(y1[c].sum())}")
# follow-up time
t0=pd.to_datetime(b.startdate_COM.str[:10]); t1=pd.to_datetime(f.startdate_COF1.str[:10],errors='coerce')
fu=((t1.reindex(free)-t0.reindex(free)).dt.days/365.25)[case1|ctrl1]
print(f" BL->F1 interval years: median {fu.median():.2f} IQR {fu.quantile(.25):.2f}-{fu.quantile(.75):.2f} max {fu.max():.2f}")
# ---- F2 extension
gg=g.reindex(free)
y2=pd.concat({c:col(gg,c,"COF2").isin([1,11]) for c in COMP},axis=1).any(axis=1)
n2=pd.concat({c:col(gg,c,"COF2")==2 for c in COMP},axis=1).all(axis=1)
case12=case1|y2; ctrl12=ctrl1&n2&~y2
t2=pd.to_datetime(g.startdate_COF2.astype(str).str[:10],errors='coerce'); fu2=((t2.reindex(free)-t0.reindex(free)).dt.days/365.25)[ctrl12]
print(f"\nBy F2: cases {case12.sum()} controls (no at F1 & F2) {ctrl12.sum()} incidence {case12.sum()/(case12.sum()+ctrl12.sum()):.2%}; BL->F2 median {fu2.median():.2f} y")
lab=pd.DataFrame({"incident_F1":np.where(case1,1,np.where(ctrl1,0,np.nan)),"incident_F2":np.where(case12,1,np.where(ctrl12,0,np.nan)),
                  "hard_ascvd_F1":np.where(hard,1,np.where(ctrl1,0,np.nan)),"fu_years_F1":((t1.reindex(free)-t0.reindex(free)).dt.days/365.25).values},index=free)
for c in COMP: lab[f"F1_{c}"]=y1[c].astype(int).values
lab.to_csv(S+"incident_labels.csv"); print("saved",lab.shape)
# ---- intersection with multimodal data & existing splits
D="/media/data1/mbahani/dataframes/"
for sp in ["train","val","test"]:
    t=pd.read_csv(D+f"{sp}_df_v3.csv").dropna(subset=["ECG_path","retinal_image"])
    blrows=t[t.retinal_image.str.contains("_RET_IMG_BL")]
    ids=pd.Index(blrows.entity_id.unique()).intersection(lab.index)
    l=lab.loc[ids,"incident_F1"]
    print(f"{sp:5s}: participants with BL ECG+retina {blrows.entity_id.nunique()}, in incident cohort {l.notna().sum()}  cases {int((l==1).sum())} controls {int((l==0).sum())} ({(l==1).mean()/l.notna().mean():.2%})")
