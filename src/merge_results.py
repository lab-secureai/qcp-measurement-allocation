from pathlib import Path
import json
import pandas as pd
import numpy as np
from run_application_noise import aggregate,plot,NOISE
ROOT=Path(__file__).resolve().parent
D=[pd.read_csv(ROOT/path/'raw_runs.csv') for path in ('RESULTS_MAIN','RESULTS_SECOND')]
df=pd.concat(D,ignore_index=True).sort_values(['dataset','seed','noise','method']).reset_index(drop=True)
assert len(df)==900
assert len(df[['dataset','seed','noise','method']].drop_duplicates())==900
assert set(df.seed)==set(range(60,70)) and set(df.dataset)=={'breast','wine01','digits38'}
assert df.isna().sum().get('shots_total',0)==0
out=ROOT/'RESULTS_10SEEDS';out.mkdir(exist_ok=True)
df.to_csv(out/'raw_runs.csv',index=False)
g,s=aggregate(df);g.to_csv(out/'summary.csv',index=False);s.to_csv(out/'paired_savings.csv',index=False)
plot(g,out)
p=df.pivot(index=['dataset','seed','noise'],columns='method',values=['shots_total','mean_set_size','observed_coverage','ideal_containment'])
rows=[]
for noise in NOISE:
 q=p.xs(noise,level='noise')
 r=q['shots_total']['Rank+Decision'];w=q['shots_total']['Width+Width'];wd=q['shots_total']['Width+Decision']
 sav=100*(1-r/w);cal=100*(1-r/wd)
 subset=df[df.noise==noise]
 rr=subset[subset.method=='Rank+Decision'];ww=subset[subset.method=='Width+Width'];qcp=subset[subset.method=='QCP64']

 rng=np.random.default_rng(2285)
 boots=np.array([np.mean(rng.choice(sav.to_numpy(),size=len(sav),replace=True)) for _ in range(5000)])
 rows.append(dict(noise=noise,rank_shots=float(r.mean()),width_shots=float(w.mean()),
       saving_pct=float(sav.mean()),cal_saving_pct=float(cal.mean()),
       paired_saving_p025=float(np.quantile(boots,.025)),paired_saving_p975=float(np.quantile(boots,.975)),
       rank_coverage=float(rr.observed_coverage.mean()),width_coverage=float(ww.observed_coverage.mean()),
       rank_set_size=float(rr.mean_set_size.mean()),width_set_size=float(ww.mean_set_size.mean()),
       rank_containment=float(rr.ideal_containment.mean()),
       qcp64_shots=float(qcp.shots_total.mean()),qcp64_coverage=float(qcp.observed_coverage.mean()),
       qcp64_set_size=float(qcp.mean_set_size.mean()),
       n_runs=len(rr)))
summary=pd.DataFrame(rows);summary.to_csv(out/'NOISE_PAPER_TABLE.csv',index=False)
meta=dict(n_runs=30,n_conditions=150,n_method_results=900,seed_range=[60,69],
           noise_conditions={k:list(v) for k,v in NOISE.items()},dataset_status={'bank':'NOT_RUN'},
           note='Bank Marketing has no local archive, cannot download UCI from this runtime; runnable Colab pipeline included',
           source='QCP_Application_Noise_2026/RESULTS_MAIN and RESULTS_SECOND',
           method_versions='frozen QCP V4, 130 epochs, unstratified calibration/test')
(out/'RUN_METADATA.json').write_text(json.dumps(meta,indent=2))
print(summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
