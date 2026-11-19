"""Clinical impact of Model A+ vs recalibrated FRS / PREVENT in the CLSA test set: reclassification counts (<5, 5-10, >=10%)
and, at the prespecified 10% threshold, people flagged and events detected per 1,000 screened, with paired bootstrap CIs."""
import os,sys,json,numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
M='/media/data1/ravram/CLSA_validation_files/manuscript';os.chdir(M)
lg=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
te=pd.read_parquet('results_v5/test_predictions_wcr.parquet').drop(columns=['DeepECG-SSL ECG-only']).merge(pd.read_parquet('results_v5/test_predictions_simple.parquet'),on='entity_id')
y=te.incident_F2_primary.values.astype(int);AP=te['DeepECG-SSL + age + sex + hypertension'].values
sc=pd.read_parquet('data_v4/scores_v4.parquet');dv=pd.concat([sc[sc.split==s] for s in ['train','val']])
R={}
for nm,c in [('FRS','frs'),('PREVENT','prevent_cvd_winsor'),('PCE','pce'),('SCORE2/OP','score2')]:
 m=LogisticRegression(C=1e6).fit(lg(dv[c].values).reshape(-1,1),dv.incident_F2_primary.astype(int));R[nm]=m.predict_proba(lg(te[c].values).reshape(-1,1))[:,1]
cat=lambda p:np.digitize(p,[.05,.10])
def stats(y,a,o):
 ca,co=cat(a),cat(o);up,dn=ca>co,ca<co;e,ne=y==1,y==0
 fa,fo=a>=.10,o>=.10;n=len(y)
 return {'nri_events':up[e].mean()-dn[e].mean(),'nri_nonevents':dn[ne].mean()-up[ne].mean(),
  'events_up':int(up[e].sum()),'events_down':int(dn[e].sum()),'nonevents_up':int(up[ne].sum()),'nonevents_down':int(dn[ne].sum()),'reclassified_pct':(ca!=co).mean()*100,
  'flagged_per1000_A':fa.mean()*1000,'flagged_per1000_O':fo.mean()*1000,'flag_diff_per1000':(fa.mean()-fo.mean())*1000,
  'events_per1000_A':(fa&e).sum()/n*1000,'events_per1000_O':(fo&e).sum()/n*1000,'event_diff_per1000':((fa&e).sum()-(fo&e).sum())/n*1000,
  'fp_diff_per1000':((fa&ne).sum()-(fo&ne).sum())/n*1000,'nnf_A':fa.sum()/max((fa&e).sum(),1),'nnf_O':fo.sum()/max((fo&e).sum(),1),
  'events_only_A':int((fa&~fo&e).sum()),'events_only_O':int((fo&~fa&e).sum()),'nonevents_only_A':int((fa&~fo&ne).sum()),'nonevents_only_O':int((fo&~fa&ne).sum())}
rng=np.random.default_rng(2026);B=[rng.integers(0,len(y),len(y)) for _ in range(1000)]
out={'n':len(y),'events':int(y.sum())}
for nm,o in R.items():
 p=stats(y,AP,o);bt=pd.DataFrame([stats(y[b],AP[b],o[b]) for b in B])
 out[nm]={k:(round(float(v),4),[round(float(x),4) for x in np.percentile(bt[k],[2.5,97.5])]) for k,v in p.items()}
json.dump(out,open('/volume/CLSA/outputs/revision/clinical_impact.json','w'),indent=1)
for nm in R:
 print('==',nm);[print(f'  {k:22s} {v[0]:>9} {v[1]}') for k,v in out[nm].items()]
