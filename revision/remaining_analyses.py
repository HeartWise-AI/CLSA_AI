"""T2/T4/T6 on frozen saved predictions; only aggregate outputs leave the NAS."""
from pathlib import Path
import hashlib,json,platform
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr,spearmanr,rankdata,beta
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score
from sklearn.linear_model import LogisticRegression
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__import__('os').environ.get('CLSA_WORKSPACE', '/volume/CLSA'))
NAS=Path('/media/data1/ravram/CLSA_validation_files/manuscript')
OUT=ROOT/'outputs/revision'
AP='DeepECG-SSL + age + sex + hypertension'
BP='Age + sex + hypertension'
PAIRS=[('A+ versus A++',AP,'DeepECG-SSL + retina + age + sex + hypertension'),('Bedside-only versus B',BP,'Retina + age + sex + hypertension')]
ROC={'Model A+':AP,'PREVENT':'prevent_cvd_winsor','SCORE2/OP':'score2','PCE':'pce','FRS':'frs'}
S3={'Joint ECG + 11 clinical':'DeepECG-SSL + clinical','Retina + 11 clinical':'Retina+clin','Fused ECG/retina + clinical':'Fusion: DeepECG-SSL + retina','PREVENT':'prevent_cvd_winsor','SCORE2/OP':'score2','PCE':'pce','FRS':'frs'}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def roc_grid(y,p,grid):
    f,t,_=roc_curve(y,p,drop_intermediate=False)
    u,idx=np.unique(f,return_index=True)
    # Upper endpoint of each vertical ROC segment, including FPR=0.
    return np.interp(grid,u,np.maximum.reduceat(t,idx))

def agreement(a,b,t):
    delta=b-a;aa=a>=t;bb=b>=t
    return {'mean_risk_difference_second_minus_first':np.mean(delta),'mean_absolute_risk_difference':np.mean(abs(delta)),
        'median_absolute_risk_difference':np.median(abs(delta)),'p95_absolute_risk_difference':np.quantile(abs(delta),.95),
        'pearson_r':pearsonr(a,b).statistic,'spearman_rho':spearmanr(a,b).statistic,
        'decision_agreement':np.mean(aa==bb),'second_only_flagged':np.mean(~aa & bb),'first_only_flagged':np.mean(aa & ~bb)}

def delong_ci(y,p):
    """Same midrank DeLong variance convention as the recorded original analysis."""
    pos,neg=p[y==1],p[y==0];m,n=len(pos),len(neg)
    ranks=rankdata(np.concatenate([pos,neg]))
    auc=ranks[:m].sum()/m/n-(m+1)/2/n
    v01=(ranks[:m]-rankdata(pos))/n
    v10=1-(ranks[m:]-rankdata(neg))/m
    se=np.sqrt(np.var(v01,ddof=1)/m+np.var(v10,ddof=1)/n)
    return auc,auc-1.96*se,auc+1.96*se

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths={k:NAS/'results_v5'/v for k,v in [('wcr','test_predictions_wcr.parquet'),('simple','test_predictions_simple.parquet')]}
    w=pd.read_parquet(paths['wcr']);s=pd.read_parquet(paths['simple'])
    assert w.entity_id.is_unique and s.entity_id.is_unique
    assert set(w.entity_id)==set(s.entity_id)
    d=w.drop(columns=['DeepECG-SSL ECG-only']).merge(s,on='entity_id',validate='one_to_one',sort=False)
    assert len(d)==3906 and d.incident_F2_primary.sum()==394
    y=d.incident_F2_primary.to_numpy(dtype=int)
    needed=set(ROC.values())|set(S3.values())|{v for _,a,b in PAIRS for v in (a,b)}
    assert np.isfinite(d[list(needed)]).all().all()
    assert ((d[list(needed)]>=0)&(d[list(needed)]<=1)).all().all()
    plan={'tasks':['T2','T4','T6'],'outcome':'incident_F2_primary','n':len(d),'events':int(y.sum()),
        'seed':2026,'bootstrap_draws':1000,'sampling':'ordinary paired participant bootstrap with replacement; one-class draws omitted',
        'confidence_interval':'percentile 2.5/97.5%; ROC bands are pointwise, not simultaneous',
        'scalar_auroc_ci':'DeLong intervals in figure to retain original manuscript convention; bootstrap intervals also exported',
        'zero_discordance':'Also report exact binomial decision-agreement CI; percentile bootstrap degenerates when no discordances are observed',
        'roc_grid':{'start':0,'stop':1,'points':201,'duplicate_fpr':'maximum TPR at each duplicate FPR then linear interpolation'},
        'thresholds':{'primary':.10,'exploratory':[.05,.075,.20]},'roc_columns':ROC,'s3_columns':S3,'agreement_pairs':PAIRS,
        'input_sha256':{k:sha(v) for k,v in paths.items()},
        'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
        'disclosure_policy':'CLSA Publication Policy v4.0 (2025-03-27), section 2.6: suppress cells <6 and prevent complementary disclosure. Agreement CSV is an internal research-team diagnostic; manuscript suppresses sparse decision-discordance results. Histogram bins <6 are omitted.',
        'policy_url':'https://www.clsa-elcv.ca/wp-content/uploads/2023/06/CLSA_PubPolicyApprvUsers_v4.0_2025Mar27.pdf'}
    (OUT/'analysis_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    rng=np.random.default_rng(2026)
    bs=[rng.integers(0,len(y),len(y)) for _ in range(1000)]
    bs=[b for b in bs if 0<y[b].sum()<len(b)]
    grid=np.linspace(0,1,201);curves=[];aucs=[];delong_rows=[]
    for name,c in ROC.items():
        p=d[c].to_numpy();boot=np.array([roc_grid(y[b],p[b],grid) for b in bs]);lo,hi=np.quantile(boot,[.025,.975],axis=0)
        point=roc_grid(y,p,grid)
        curves.extend({'model':name,'fpr':float(f),'tpr':float(t),'lower':float(l),'upper':float(h)} for f,t,l,h in zip(grid,point,lo,hi))
        bauc=np.array([roc_auc_score(y[b],p[b]) for b in bs]);al,ah=np.quantile(bauc,[.025,.975]);aucs.append({'model':name,'auroc':roc_auc_score(y,p),'lower':al,'upper':ah})
        da,dl,dh=delong_ci(y,p);delong_rows.append({'model':name,'auroc':da,'lower':dl,'upper':dh})
    roc=pd.DataFrame(curves);roc.to_csv(OUT/'roc_bands.csv',index=False);pd.DataFrame(aucs).to_csv(OUT/'roc_auroc_ci.csv',index=False)
    pd.DataFrame(delong_rows).to_csv(OUT/'roc_auroc_delong_ci.csv',index=False)
    rows=[]
    for name,c in S3.items():
        p=d[c].to_numpy();v=average_precision_score(y,p);bv=np.array([average_precision_score(y[b],p[b]) for b in bs]);lo,hi=np.quantile(bv,[.025,.975]);rows.append({'model':name,'column':c,'auprc':v,'lower':lo,'upper':hi,'bootstrap_valid':len(bs)})
    ci=pd.DataFrame(rows);ci.to_csv(OUT/'s3_auprc_ci.csv',index=False)
    for k,v in zip(list(S3)[:3],[.286,.223,.256]):assert abs(float(ci.set_index('model').loc[k,'auprc'])-v)<.0005
    # Check the existing comparator CIs reproduce with the original seed and estimator.
    expected={'PREVENT':(.191,.254),'SCORE2/OP':(.184,.247),'PCE':(.189,.252),'FRS':(.168,.228)}
    for k,(lo,hi) in expected.items():
        r=ci.set_index('model').loc[k];assert round(r.lower,3)==lo and round(r.upper,3)==hi,(k,r.lower,r.upper)
    rows=[];bins=[]
    fig,ax=plt.subplots(1,2,figsize=(10,4.5))
    for j,(name,c1,c2) in enumerate(PAIRS):
        a,b=d[c1].to_numpy(),d[c2].to_numpy()
        for t in [.10,.05,.075,.20]:
            point=agreement(a,b,t);boot=[agreement(a[z],b[z],t) for z in bs]
            for key,value in point.items():
                lo,hi=np.quantile([r[key] for r in boot],[.025,.975]);rows.append({'comparison':name,'threshold':t,'status':'primary' if t==.10 else 'exploratory','metric':key,'estimate':value,'lower':lo,'upper':hi,'bootstrap_valid':len(bs)})
            k=int(np.sum((a>=t)==(b>=t)));n=len(a)
            lo=0 if k==0 else beta.ppf(.025,k,n-k+1)
            hi=1 if k==n else beta.ppf(.975,k+1,n-k)
            rows.append({'comparison':name,'threshold':t,'status':'primary' if t==.10 else 'exploratory','metric':'decision_agreement_exact_binomial','estimate':k/n,'lower':lo,'upper':hi,'bootstrap_valid':None})
        h,xe,ye=np.histogram2d(a,b,bins=np.linspace(0,1,51))
        for ix,iy in zip(*np.nonzero(h>=6)):bins.append({'comparison':name,'first_risk_bin_left':xe[ix],'second_risk_bin_left':ye[iy],'count':int(h[ix,iy])})
        im=ax[j].pcolormesh(xe,ye,np.ma.masked_less(h.T,6),cmap='viridis',shading='flat');fig.colorbar(im,ax=ax[j],label='Participants per displayed risk bin (≥6)')
        ax[j].plot([0,1],[0,1],'k--',lw=.7);ax[j].axhline(.1,color='gray',lw=.5);ax[j].axvline(.1,color='gray',lw=.5)
        ax[j].set(title=name,xlabel='First model risk',ylabel='Second model risk',xlim=(0,1),ylim=(0,1))
    fig.suptitle('Risk bins with fewer than 6 participants are not displayed',fontsize=10)
    fig.tight_layout();fig.savefig(OUT/'prediction_agreement.png',dpi=200);plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/'prediction_agreement.csv',index=False);pd.DataFrame(bins).to_csv(OUT/'prediction_agreement_bins.csv',index=False)
    # Recreate all three panels from the existing source, changing ROC bands only.
    sc=pd.read_parquet(NAS/'data_v4/scores_v4.parquet');dv=sc[sc.split.isin(['train','val'])]
    def lg(p):return np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
    for name,c in list(ROC.items())[1:]:
        m=LogisticRegression(C=1e6).fit(lg(dv[c].to_numpy()).reshape(-1,1),dv.incident_F2_primary.astype(int));d[name+' recal']=m.predict_proba(lg(d[c].to_numpy()).reshape(-1,1))[:,1]
    colors={'Model A+':'#b2182b','PREVENT':'#333333','SCORE2/OP':'#666666','PCE':'#999999','FRS':'#555555'}
    # Sized for a ~16.5-cm manuscript column: retain readable labels at print scale.
    plt.rcParams.update({'font.size':9})
    fig=plt.figure(figsize=(7,6.7))
    gs=fig.add_gridspec(2,2,height_ratios=[1.1,1])
    ax=[fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]
    for name,c in ROC.items():
        f,t,_=roc_curve(y,d[c]);g=roc[roc.model==name];a=next(r for r in delong_rows if r['model']==name)
        ax[0].plot(f,t,color=colors[name],lw=2 if name=='Model A+' else 1,label=f"{name}: {a['auroc']:.3f} ({a['lower']:.3f}–{a['upper']:.3f})")
        ax[0].fill_between(g.fpr,g.lower,g.upper,color=colors[name],alpha=.12)
    ax[0].plot([0,1],[0,1],'k:',lw=.8);ax[0].set(xlabel='1 − specificity',ylabel='Sensitivity',title='95% pointwise ROC bands');ax[0].legend(fontsize=8,loc='lower right')
    for name,c in [('Model A+',AP),('PREVENT','PREVENT recal'),('FRS','FRS recal')]:
        g=d.groupby(pd.qcut(d[c],10,duplicates='drop'),observed=True);ax[1].plot(g[c].mean(),g.incident_F2_primary.mean(),'o-',color=colors[name],label=name+(' (recalibrated)' if name!='Model A+' else ''))
    ax[1].plot([0,.4],[0,.4],'k:',lw=.8);ax[1].set(xlabel='Predicted risk by Follow-up 2',ylabel='Observed risk by Follow-up 2');ax[1].legend(fontsize=8)
    ths=np.arange(.02,.205,.01)
    ax[2].plot(ths,[y.mean()-(1-y.mean())*t/(1-t) for t in ths],color='.7',label='Treat all');ax[2].axhline(0,color='k',lw=.6,label='Treat none')
    for name,c in [('Model A+',AP)]+[(k,k+' recal') for k in list(ROC)[1:]]:
        p=d[c].to_numpy();nb=[np.mean((p>=t)&(y==1))-np.mean((p>=t)&(y==0))*t/(1-t) for t in ths]
        ax[2].plot(ths,nb,color=colors[name],label=name)
    ax[2].set(ylim=(-.01,.07),xlabel='Threshold probability',ylabel='Net benefit');ax[2].legend(fontsize=8)
    for i,a in enumerate(ax):a.text(-.12,1.02,'ABC'[i],transform=a.transAxes,fontsize=14,fontweight='bold')
    fig.tight_layout();fig.savefig(OUT/'Figure2_CLSA_ROC_calibration_DCA.png',dpi=220);fig.savefig(OUT/'Figure2_CLSA_ROC_calibration_DCA.pdf');plt.close(fig)
    summary={'status':'complete_computation_pending_document_insertion','n':len(y),'events':int(y.sum()),'bootstrap_valid':len(bs),'comparator_intervals_reproduced':True,'tasks':['T2','T4','T6']}
    (OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
