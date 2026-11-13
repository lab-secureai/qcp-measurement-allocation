import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
repo=Path(__file__).resolve().parents[1]
src=repo/'results/factorial/unstratified/factorial_summary.csv'
out=repo/'figures/generated'
out.mkdir(parents=True,exist_ok=True)
D=pd.read_csv(src).set_index('method')
fig,ax=plt.subplots(figsize=(6.9,3.55))
order=['R+D','W+D','R+W','W+W','Wbudget+D']
labels={'R+D':'Rank + decision','W+D':'Width + decision','R+W':'Rank + width','W+W':'Width + width','Wbudget+D':'Width-first + decision\n(matched budget)'}
shifts={'R+D':(5,-17),'W+D':(7,9),'R+W':(-30,-18),'W+W':(-100,8),'Wbudget+D':(9,11)}
for key in order:
 row=D.loc[key]
 x=row['mean_shots']/1000;y=row['mean_set_size']
 ax.scatter(x,y,s=44,marker='o' if key!='R+D' else 'D',zorder=3)
 ax.annotate(labels[key],(x,y),xytext=shifts[key],textcoords='offset points',fontsize=8)
ax.set_xlabel('Mean calibration and test shots per run (thousands)')
ax.set_ylabel('Mean prediction set size')
ax.set_xlim(70,160);ax.set_ylim(1.202,1.255)
ax.grid(alpha=.18)
fig.tight_layout(pad=1.2)
for suffix in ['pdf','png']:
 fig.savefig(out/('factorial_tradeoff.'+suffix), dpi=200)
plt.close(fig)
