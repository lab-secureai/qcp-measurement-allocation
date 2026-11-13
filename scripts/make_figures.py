from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
OUT=ROOT/'figures'/'generated'
OUT.mkdir(parents=True,exist_ok=True)


r=pd.read_csv(RESULTS/'factorial/unstratified/factorial_all_runs.csv')
order=['R+D','W+D','R+W','W+W']
g=r.groupby('method').agg(shots=('total_shots','mean'),size=('mean_set_size','mean')).reindex(order)
fig,ax=plt.subplots(figsize=(7.5,4.2))
ax.bar(np.arange(len(order)),g.shots.values)
ax.set_xticks(np.arange(len(order)),['Rank + Decision','Width + Decision','Rank + Width','Width + Width'])
ax.set_ylabel('Mean calibration + test shots')
ax.set_title('Factorial measurement allocation comparison')
ax.tick_params(axis='x',rotation=12)
fig.tight_layout();fig.savefig(OUT/'factorial_shots.png',dpi=180);fig.savefig(OUT/'factorial_shots.pdf');plt.close(fig)


r=pd.read_csv(RESULTS/'noise/raw_runs.csv')
p=r.pivot(index=['dataset','seed','noise'],columns='method',values='shots_total')
z=100*(1-p['Rank+Decision']/p['Width+Width'])
sv=z.groupby(level='noise').mean()
noises=['ideal','readout_2pct','gate_depolarizing','combined_mild','combined_strong']
fig,ax=plt.subplots(figsize=(7.6,4.3))
ax.plot(range(len(noises)),[sv[n] for n in noises],marker='o')
ax.set_xticks(range(len(noises)),['Ideal','Readout 2%','Gate depol.','Combined mild','Combined strong'])
ax.set_ylabel('Mean paired shot saving (%)')
ax.set_title('Measurement savings under stationary noise')
ax.tick_params(axis='x',rotation=15)
fig.tight_layout();fig.savefig(OUT/'noise_paired_savings.png',dpi=180);fig.savefig(OUT/'noise_paired_savings.pdf');plt.close(fig)


bank=pd.read_csv(RESULTS/'bank_marketing/raw_runs.csv')
q=bank[bank.method=='Rank+Decision'].groupby('noise').agg(
    marginal=('observed_coverage','mean'),positive=('conditional_positive_coverage','mean')).reindex(noises)
fig,ax=plt.subplots(figsize=(7.6,4.4))
x=np.arange(len(noises));w=.36
ax.bar(x-w/2,q.marginal.values,width=w,label='Marginal coverage')
ax.bar(x+w/2,q.positive.values,width=w,label='Positive-class coverage')
ax.set_xticks(x,['Ideal','Readout 2%','Gate depol.','Combined mild','Combined strong'])
ax.set_ylim(0,1.05)
ax.set_ylabel('Observed coverage')
ax.set_title('Bank Marketing: observed coverage by noise regime')
ax.tick_params(axis='x',rotation=15)
ax.legend();fig.tight_layout();fig.savefig(OUT/'bank_coverage.png',dpi=180);fig.savefig(OUT/'bank_coverage.pdf');plt.close(fig)
print('Wrote figures to',OUT)
