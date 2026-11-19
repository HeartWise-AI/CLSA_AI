"""MHI operating characteristics at the prespecified 10% high-risk threshold (5%, 7.5%, 20% exploratory).
Frozen CLSA coefficients (as transported) and after 5-fold cross-fitted logistic recalibration at MHI, as in Supplementary Table S16."""
from pathlib import Path
import json
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
ROOT=Path(__import__('os').environ.get('CLSA_WORKSPACE', '/volume/CLSA'));NAS=Path('/media/data1/ravram/CLSA_validation_files/manuscript')
c=pd.read_parquet(NAS/'mhi_anyecg/cohort_labels.parquet').merge(pd.read_parquet(NAS/'mhi_anyecg/predictions.parquet'),on='MRN',validate='one_to_one')
c=c[c.ecg_ok];b=c[(c.prev=='no')&~c.unk_death&((c.y==1)|c.fu_ecg_6y)].reset_index(drop=True);assert len(b)==21647 and b.y.sum()==2190
P=json.loads((ROOT/'DeepECG_Docker/clsa_cvd/model_params.json').read_text());A=P['model_ecg_age_sex_external'];S=P['age_sex_reference']
b['pA']=expit(A['intercept']+A['b_ecg']*b.wcr+A['b_age']*b.age+A['b_female']*b.female)
b['pS']=expit(S['intercept']+S['b_age']*b.age+S['b_female']*b.female)
y=b.y.values.astype(int);lg=lambda p:np.log(p/(1-p))
assert abs(roc_auc_score(y,b.pA)-.761)<.0005 and abs(roc_auc_score(y,b.pS)-.728)<.0005 and abs(b.pA.mean()-.164)<.0005
for k in ['pA','pS']:  # same cross-fitting as extra_analyses3.py (S16, Figure S2)
 p=b[k].values;rc=np.zeros(len(p))
 for tr,ts in StratifiedKFold(5,shuffle=True,random_state=0).split(p,y):
  rc[ts]=LogisticRegression(C=1e6).fit(lg(p[tr]).reshape(-1,1),y[tr]).predict_proba(lg(p[ts]).reshape(-1,1))[:,1]
 b[k+'_rc']=rc
def ops(y,p,t):
 f=p>=t;tp=(f&(y==1)).sum();fp=(f&(y==0)).sum();fn=(~f&(y==1)).sum();tn=(~f&(y==0)).sum();n=len(y)
 return {'high_risk_pct':f.mean(),'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'ppv':tp/(tp+fp) if tp+fp else np.nan,'npv':tn/(tn+fn) if tn+fn else np.nan,
         'net_benefit':(tp-fp*t/(1-t))/n,'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn)}
rng=np.random.default_rng(2026);B=[rng.integers(0,len(y),len(y)) for _ in range(1000)]
rows=[]
for model,k in [('DeepECG-SSL + age + sex','pA'),('Age + sex','pS')]:
 for cal,col in [('as transported','%s'%k),('recalibrated at MHI (cross-fitted)',k+'_rc')]:
  p=b[col].values
  for t in [.05,.075,.10,.20]:
   o=ops(y,p,t);bt=pd.DataFrame([ops(y[i],p[i],t) for i in B])
   r={'model':model,'calibration':cal,'threshold':t,'status':'prespecified' if t==.10 else 'exploratory'}
   for m in ['high_risk_pct','sensitivity','specificity','ppv','npv','net_benefit']:
    r[m]=o[m];r[m+'_lo'],r[m+'_hi']=np.nanpercentile(bt[m],[2.5,97.5])
   r.update({x:o[x] for x in ['tp','fp','fn','tn']});rows.append(r)
d=pd.DataFrame(rows);out=ROOT/'outputs/revision/mhi_threshold_operating_characteristics.csv';d.to_csv(out,index=False)
pd.set_option('display.width',250)
f=lambda r,m,s=100:f"{r[m]*s:.1f} ({r[m+'_lo']*s:.1f}–{r[m+'_hi']*s:.1f})"
print(pd.DataFrame([{'model':r.model[:9],'cal':r.calibration[:5],'t':r.threshold,**{m:f(r,m) for m in ['high_risk_pct','sensitivity','specificity','ppv','npv']},'NB':f"{r.net_benefit:.4f}"} for _,r in d.iterrows()]).to_string(index=False))
