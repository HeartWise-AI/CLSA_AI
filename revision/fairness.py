"""T1: source-verified self-identified cultural/racial groups; disclosure-safe output."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.special import expit
from scipy.optimize import minimize,brentq
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__import__('os').environ.get('CLSA_WORKSPACE', '/volume/CLSA'));OUT=ROOT/'outputs/revision';NAS=Path('/media/data1/ravram/CLSA_validation_files/manuscript')
BL=Path('/media/data1/mbahani/CLSA/CLSA_ALL/CLSA_Data/CLSA/2401002_MontrealHeartInstitute_RAvram_BL/2401002_MontrealHeartInstitute_RAvram_Baseline_CoPv7.csv')
DIC=BL.with_name(BL.stem+'-dictionary.xlsx')
POLICY='https://www.clsa-elcv.ca/wp-content/uploads/2023/06/CLSA_PubPolicyApprvUsers_v4.0_2025Mar27.pdf'

def calibration(y,p):
 x=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
 m=minimize(lambda b: np.mean(np.logaddexp(0,b[0]+b[1]*x)-y*(b[0]+b[1]*x)),[0,1],jac=lambda b:np.array([np.mean(expit(b[0]+b[1]*x)-y),np.mean((expit(b[0]+b[1]*x)-y)*x)]),method='BFGS')
 if not m.success or not np.isfinite(m.x).all():return np.nan,np.nan
 return float(m.x[1]),float(brentq(lambda b:np.mean(expit(b+x))-np.mean(y),-50,50))

def values(y,p):
 flag=p>=.1;tp=np.sum(flag&(y==1));fn=np.sum(~flag&(y==1));fp=np.sum(flag&(y==0));tn=np.sum(~flag&(y==0))
 slope,citl=calibration(y,p)
 def rate(a,b):return a/(a+b) if a+b else np.nan
 return {'auroc':roc_auc_score(y,p),'auprc':average_precision_score(y,p),'brier':np.mean((y-p)**2),'calibration_slope':slope,'calibration_in_large':citl,'sensitivity':rate(tp,fn),'specificity':rate(tn,fp),'ppv':rate(tp,fp),'npv':rate(tn,fn)}

def main():
 var=pd.read_excel(DIC,sheet_name='Variables');cat=pd.read_excel(DIC,sheet_name='Categories')
 cols=[c for c in pd.read_csv(BL,nrows=0).columns if c.startswith('SDC_CULT_') and c.endswith('_COM') and c not in ['SDC_CULT_DK_NA_COM','SDC_CULT_REFUSED_COM','SDC_CULT_OTSP_COM']]
 raw=pd.read_csv(BL,usecols=['entity_id']+cols).set_index('entity_id');assert raw.index.is_unique
 te=pd.read_parquet(NAS/'results_v5/test_predictions_wcr.parquet')[['entity_id','incident_F2_primary']]
 sp=pd.read_parquet(NAS/'results_v5/test_predictions_simple.parquet')[['entity_id','DeepECG-SSL + age + sex + hypertension']]
 te=te.merge(sp,on='entity_id',validate='one_to_one');joined=te.join(raw,on='entity_id',validate='one_to_one')
 assert joined.entity_id.is_unique and len(joined)==3906
 rows=[];provenance=[]
 for col in cols:
  desc=var[var.name==col].iloc[0];opts=cat[cat.variable==col]
  mapping=dict(zip(opts.name.astype(str),opts.label.astype(str)));assert mapping.get('1')=='Yes' and mapping.get('0')=='No'
  label=str(desc.label).replace("Participant's cultural background ",'')
  provenance.append({'column':col,'label':label,'yes_code':1,'no_code':0,'source':'Baseline CoPv7 dictionary'})
  code=pd.to_numeric(joined[col],errors='coerce');g=joined[code==1]
  y=g.incident_F2_primary.to_numpy(dtype=int);p=g['DeepECG-SSL + age + sex + hypertension'].to_numpy()
  # Suppress the whole row when event/non-event cells are sparse, so totals cannot reveal them.
  if len(g)<6 or y.sum()<6 or (len(g)-y.sum())<6:
   rows.append({'group':label,'n':'withheld','events':'withheld','metric':'all','estimate':'withheld','lower':'withheld','upper':'withheld','reason':'Insufficient disclosable event/non-event cells; no fairness conclusion possible'});continue
  flag=p>=.1;tp=int(np.sum(flag&(y==1)));fn=int(np.sum(~flag&(y==1)));fp=int(np.sum(flag&(y==0)));tn=int(np.sum(~flag&(y==0)))
  safe={'sensitivity':min(tp,fn)>=6,'specificity':min(tn,fp)>=6,'ppv':min(tp,fp)>=6,'npv':min(tn,fn)>=6}
  point=values(y,p);rng=np.random.default_rng(2026);boot=[]
  for _ in range(1000):
   b=rng.integers(0,len(y),len(y))
   if 0<y[b].sum()<len(b):boot.append(values(y[b],p[b]))
  for metric,v in point.items():
   if metric in safe and not safe[metric]:
    rows.append({'group':label,'n':len(g),'events':int(y.sum()),'metric':metric,'estimate':'withheld','lower':'withheld','upper':'withheld','reason':'Suppressed sparse confusion cells; intervals also withheld'});continue
   vals=np.array([r[metric] for r in boot]);vals=vals[np.isfinite(vals)]
   if len(vals)<950 or not np.isfinite(v):
    rows.append({'group':label,'n':len(g),'events':int(y.sum()),'metric':metric,'estimate':'not estimable','lower':'not estimable','upper':'not estimable','reason':'Unstable calibration/bootstrap estimation'});continue
   lo,hi=np.quantile(vals,[.025,.975]);rows.append({'group':label,'n':len(g),'events':int(y.sum()),'metric':metric,'estimate':v,'lower':lo,'upper':hi,'reason':'Exploratory; groups may overlap; no causal or equality inference'})
 pd.DataFrame(rows).to_csv(OUT/'fairness_disclosure_review.csv',index=False)
 report={'task':'T1','status':'feasibility_and_estimable_group_metrics_computed','threshold':.1,'bootstrap':1000,'seed':2026,'groups':'Each original SDC_CULT_* yes indicator; multi-response identities can overlap. Missing/nonbinary answers are not recoded as No.','policy':POLICY,'rule':'Suppress cells <6, rates relying on sparse confusion cells and whole groups with sparse event/non-event cells. No table of mutually exclusive totals is emitted, avoiding complementary disclosure.','dictionary':str(DIC),'variables':provenance,'interpretation':'This sample cannot establish fairness for groups with suppressed results; intervals for estimable groups are exploratory and not multiplicity-adjusted. Data-steward review required before external dissemination.'}
 (OUT/'fairness_methods.json').write_text(json.dumps(report,indent=2)+'\n')
 print('Disclosure-safe fairness report generated; no small-cell counts logged.')
if __name__=='__main__':main()
