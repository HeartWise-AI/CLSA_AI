"""Figure: clinical impact of Model A+ at the 10% threshold (per 1,000 screened) and reclassification vs recalibrated FRS (CLSA test set)."""
import json
from pathlib import Path
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
ROOT=Path(__import__('os').environ.get('CLSA_WORKSPACE', '/volume/CLSA'));d=json.loads((ROOT/'outputs/revision/clinical_impact.json').read_text())
INK,INK2,GRID='#0b0b0b','#52514e','#e4e3df';EV,NE,UP,DN='#2a78d6','#86b6ef','#eb6834','#2a78d6'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':INK2,'axes.labelcolor':INK,'xtick.color':INK2,'ytick.color':INK})
fig,(a,b)=plt.subplots(1,2,figsize=(11,4.2),gridspec_kw={'width_ratios':[1.35,1]})
# A: per 1,000 screened, flagged ≥10%
rows=[('FRS (recalibrated)','FRS','O'),('PCE (recalibrated)','PCE','O'),('SCORE2/OP (recalibrated)','SCORE2/OP','O'),('PREVENT (recalibrated)','PREVENT','O'),('Model A+','FRS','A')]
for i,(lab,k,s) in enumerate(rows):
    fl=d[k][f'flagged_per1000_{s}'][0];ev=d[k][f'events_per1000_{s}'][0];ne=fl-ev;y=len(rows)-1-i
    a.barh(y,ev,color=EV,height=.6,edgecolor='#fcfcfb',linewidth=2);a.barh(y,ne,left=ev,color=NE,height=.6,edgecolor='#fcfcfb',linewidth=2)
    a.text(fl+6,y,f'{fl:.0f} flagged · {ev:.0f} events',va='center',color=INK,fontsize=8.5,fontweight='bold' if s=='A' else None)
a.set_yticks(range(len(rows)));a.set_yticklabels([r[0] for r in rows][::-1]);a.set_xlim(0,540)
a.set_xlabel('People classified high risk (≥10%) per 1,000 screened');a.grid(axis='x',color=GRID,lw=.8);a.set_axisbelow(True)
for sp in ['top','right']:a.spines[sp].set_visible(False)
a.legend(handles=[plt.Rectangle((0,0),1,1,color=EV),plt.Rectangle((0,0),1,1,color=NE)],labels=['Developed CVD (true positive)','Did not develop CVD (false positive)'],loc='upper center',bbox_to_anchor=(.45,-.16),ncol=2,frameon=False,fontsize=8)
a.set_title('A  At the 10% threshold: same events found, fewer people flagged',loc='left',fontsize=10,color=INK,fontweight='bold')
# B: reclassification vs FRS
f=d['FRS'];cats=[('Participants with CVD\n(n = 394)','events'),('Participants without CVD\n(n = 3,512)','nonevents')]
for j,(lab,k) in enumerate(cats):
    up,dn=f[f'{k}_up'][0],f[f'{k}_down'][0];y=1-j
    b.barh(y+.17,up,color=UP,height=.32);b.barh(y-.17,dn,color=DN,height=.32)
    b.text(up+12,y+.17,f'{up:.0f} moved up',va='center',fontsize=8.5,color=INK);b.text(dn+12,y-.17,f'{dn:.0f} moved down',va='center',fontsize=8.5,color=INK)
b.set_yticks([1,0]);b.set_yticklabels([c[0] for c in cats]);b.set_xlim(0,1250);b.set_xlabel('Participants changing risk category\n(<5%, 5–10%, ≥10%), Model A+ vs FRS')
b.grid(axis='x',color=GRID,lw=.8);b.set_axisbelow(True)
for sp in ['top','right']:b.spines[sp].set_visible(False)
nri=lambda k:f"{f[k][0]:+.3f} ({f[k][1][0]:+.3f} to {f[k][1][1]:+.3f})"
b.set_title('B  Reclassification versus FRS',loc='left',fontsize=10,color=INK,fontweight='bold')
plt.tight_layout();out=ROOT/'outputs/revision/Figure_clinical_impact.png';plt.savefig(out,dpi=300,facecolor='white');plt.savefig(out.with_suffix('.pdf'),facecolor='white');print(out)
