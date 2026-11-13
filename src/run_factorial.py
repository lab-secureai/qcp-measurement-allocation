import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
import experiment_width_matched as exp
import qcp_rank_v2 as base
from v3_reference import test_with_spending

METHODS=['R+D','W+D','R+W','W+W','Wbudget+D']

def run_one(ds, seed, epochs=130):
    run=exp.init_run(ds,seed,epochs)
    yc=run['yc']; sc=run['cal']; st=run['test']; rho=run['rho']
    rank_c=run['old']['shots']; rank_q=run['qrank']
    width_q,width_c,width_stage=exp.width_cal_threshold(yc,sc,0.18)
    budget_q,budget_c,budget_stage=exp.width_cal_budgeted(yc,sc,rank_c,0.)
    sets_rd,td,_=test_with_spending(rank_q,st,rho)
    sets_wd,twd,_=test_with_spending(width_q,st,rho)
    sets_rw,trw,_=exp.width_only_test(rank_q,st,0.12,rho)
    sets_ww,tww,_=exp.width_only_test(width_q,st,0.12,rho)
    sets_bd,tbd,_=test_with_spending(budget_q,st,rho)
    items=[('R+D',sets_rd,rank_c,td,rank_q),('W+D',sets_wd,width_c,twd,width_q),
         ('R+W',sets_rw,rank_c,trw,rank_q),('W+W',sets_ww,width_c,tww,width_q),
         ('Wbudget+D',sets_bd,budget_c,tbd,budget_q)]
    rows=[exp.row(run,name,sets,cs,ts,q) for name,sets,cs,ts,q in items]

    details=[]
    for name,sets,cs,ts,q in items:
        for i in range(len(run['yv'])):
            details.append({'dataset':ds,'seed':seed,'sample_index':i,'method':name,
                'covered':int(sets[i,run['yv'][i]]),'set_size':int(sets[i].sum()),
                'set_mask':int(sets[i,0])+2*int(sets[i,1])})
    return rows,details

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=50)
    ap.add_argument('--n-seeds',type=int,default=10);ap.add_argument('--epochs',type=int,default=130)
    ap.add_argument('--split-policy',choices=['random_holdout','original_stratified'],default='random_holdout')
    args=ap.parse_args()
    exp.SPLIT_POLICY=args.split_policy
    root=Path(__file__).parent; out=root/('RESULTS_UNSTRATIFIED' if args.split_policy=='random_holdout' else 'RESULTS_STRATIFIED');out.mkdir(exist_ok=True)
    start=time.time();rows=[];details=[]
    for ds in exp.DATASETS:
        for seed in range(args.start,args.start+args.n_seeds):
            a,b=run_one(ds,seed,args.epochs);rows.extend(a);details.extend(b)
            print(f'completed {ds} {seed}, elapsed {time.time()-start:.1f}s',flush=True)
    raw=pd.DataFrame(rows); det=pd.DataFrame(details)
    raw.to_csv(out/'factorial_all_runs.csv',index=False);det.to_csv(out/'factorial_sample_details.csv',index=False)
    agg=raw.groupby('method',as_index=False).agg(mean_shots=('total_shots','mean'),
        mean_cal_shots=('cal_shots','mean'),mean_test_shots=('test_shots','mean'),
        mean_set_size=('mean_set_size','mean'),mean_coverage=('coverage','mean'),
        mean_ideal_containment=('ideal_set_containment','mean'))
    agg.to_csv(out/'factorial_summary.csv',index=False)
    p=raw.pivot(index=['dataset','seed'],columns='method',values=['total_shots','cal_shots','test_shots','mean_set_size','coverage'])
    def cmp(a,b):
        return dict(a=a,b=b,saving_pct=float(100*(1-p['total_shots'][a]/p['total_shots'][b]).mean()),
          mean_shot_delta=float((p['total_shots'][a]-p['total_shots'][b]).mean()),
          mean_set_delta=float((p['mean_set_size'][a]-p['mean_set_size'][b]).mean()),
          mean_coverage_delta=float((p['coverage'][a]-p['coverage'][b]).mean()))
    contrasts=[cmp('R+D','W+D'),cmp('R+W','W+W'),cmp('R+D','R+W'),cmp('W+D','W+W'),
       cmp('R+D','W+W'),cmp('R+D','Wbudget+D')]

    idx=p.index;grp=[np.flatnonzero(idx.get_level_values(0)==ds) for ds in exp.DATASETS];rng=np.random.default_rng(881300)
    boots=[]
    for _ in range(3000):
        sampled=np.concatenate([rng.choice(grp[g],size=len(grp[g]),replace=True) for g in rng.integers(0,3,size=3)])
        boots.append([100*(1-(p['total_shots'][a].to_numpy()[sampled]/p['total_shots'][b].to_numpy()[sampled])).mean() for a,b in [('R+D','W+D'),('R+D','W+W'),('R+D','Wbudget+D')]])
    bs=np.asarray(boots)
    for pair,dist in zip([('R+D','W+D'),('R+D','W+W'),('R+D','Wbudget+D')],bs.T):
        for r in contrasts:
            if r['a']==pair[0] and r['b']==pair[1]:r['bootstrap_95_pct']=list(np.quantile(dist,[.025,.975]));break

    refpath=Path(__file__).parent/'REFERENCE_heldout_all_runs.csv'
    check={}
    if refpath.exists() and args.split_policy=='random_holdout' and args.start==50:
        ref=pd.read_csv(refpath)
        for mine,old in [('R+D','rank_V3'),('W+W','width_selected'),('R+W','width_test_only'),('Wbudget+D','width_budget_matched')]:
            actual=raw[raw.method==mine].sort_values(['dataset','seed']).reset_index(drop=True)
            expected=ref[(ref.method==old)&(ref.seed.between(args.start,args.start+args.n_seeds-1))].sort_values(['dataset','seed']).reset_index(drop=True)
            comparison={c:float(np.max(np.abs(actual[c].to_numpy()-expected[c].to_numpy()))) for c in ['total_shots','cal_shots','test_shots','mean_set_size','coverage']}
            check[mine]=comparison
            if len(actual)!=len(expected) or any(z>1e-9 for z in comparison.values()):raise AssertionError(('reproduction mismatch',mine,comparison))
    result={'protocol':args.split_policy,'seeds':[args.start,args.start+args.n_seeds-1],
        'alpha':base.ALPHA,'beta':base.BETA,'frozen_thresholds':{'calibration_width':0.18,'test_width':0.12},
        'n_runs':len(raw)//len(METHODS),'contrasts':contrasts,
        'reproduction_checks':check,'wall_seconds':time.time()-start}
    (out/'factorial_metrics.json').write_text(json.dumps(result,indent=2))
    print(agg.to_string(index=False));print(json.dumps(result,indent=2));print('OUTPUT',out,flush=True)
if __name__=='__main__': main()
