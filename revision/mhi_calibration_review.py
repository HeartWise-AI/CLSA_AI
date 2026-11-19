from pathlib import Path
import json
import pandas as pd
from scipy.special import expit
ROOT=Path(__import__('os').environ.get('CLSA_WORKSPACE', '/volume/CLSA'));NAS=Path('/media/data1/ravram/CLSA_validation_files/manuscript')
c=pd.read_parquet(NAS/'mhi_anyecg/cohort_labels.parquet').merge(pd.read_parquet(NAS/'mhi_anyecg/predictions.parquet'),on='MRN',validate='one_to_one')
c=c[c.ecg_ok];b=c[(c.prev=='no')&~c.unk_death&((c.y==1)|c.fu_ecg_6y)].copy();assert len(b)==21647 and b.y.sum()==2190
p=json.loads((ROOT/'DeepECG_Docker/clsa_cvd/model_params.json').read_text())['model_ecg_age_sex_external']
b['pred']=expit(p['intercept']+p['b_ecg']*b.wcr+p['b_age']*b.age+p['b_female']*b.female)
r={'n':len(b),'events':int(b.y.sum()),'overall':{'predicted':float(b.pred.mean()),'observed':float(b.y.mean())}}
for name,q in [('top_decile',.9),('top_quartile',.75)]:
 g=b[b.pred>=b.pred.quantile(q)];r[name]={'predicted':float(g.pred.mean()),'observed':float(g.y.mean())}
r['finding']='Overprediction overall and in both high-risk groups; supports correcting the direction in Limitations.'
(ROOT/'outputs/revision/mhi_calibration_review.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
