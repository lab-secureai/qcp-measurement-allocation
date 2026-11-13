import argparse, json, math, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import qcp_rank_v2 as b
from v3_reference import calibrated_rho, test_with_spending

ROOT=Path(__file__).resolve().parent
SPLIT_POLICY='original_stratified'
ST=np.asarray(b.STAGES); NST=len(ST)
DATASETS=['breast','wine01','digits38']

CAL_THRESHOLDS=[0.10,0.14,0.18,0.22,0.27,0.33,0.42]
TEST_THRESHOLDS=[0.08,0.12,0.16,0.22,0.30]
WIDTH_PRIORITY_POWERS=[0.0,0.5,1.0]
TOL_SET=0.005                                                                           


def score_cp_width(count,t,err):
    lo,hi=b.cp_interval(count,t,err)
    return hi-lo


def width_cal_threshold(y, stream, threshold):


    n=len(y);stage=np.zeros(n,dtype=np.int8);active=np.ones(n,dtype=bool)
    e=b.BETA_CAL/(n*NST)
    for j,t in enumerate(ST):
        ix=np.flatnonzero(active)
        if not len(ix):break
        c=stream[ix,t-1]
        w=score_cp_width(c,t,e)
        stage[ix]=j+1
        active[ix[w<=threshold]]=False
    qhi,_=b.quantile_focused_upper(y,stream,stage)
    shots=int(np.sum(ST[stage-1]))
    return qhi,shots,stage


def width_cal_budgeted(y,stream, budget, priority_power=0.):


    n=len(y); stage=np.ones(n,dtype=np.int8)
    cost=int(ST[0]*n);budget=max(cost,min(int(budget),int(ST[-1]*n)))
    e=b.BETA_CAL/(n*NST)
    width=score_cp_width(stream[:,ST[0]-1],int(ST[0]),e)
    while True:
        can=np.flatnonzero(stage<NST)
        if not len(can):break
        extra=ST[stage[can]]-ST[stage[can]-1]
        can=can[extra<=budget-cost]
        if not len(can):break

        next_cost=ST[stage[can]]-ST[stage[can]-1]
        priority=width[can]/np.power(next_cost,priority_power)
        idx=can[np.argmax(priority)]
        new_j=stage[idx]
        extra=int(ST[new_j]-ST[new_j-1])
        stage[idx]+=1;cost+=extra
        t=int(ST[stage[idx]-1]);c=int(stream[idx,t-1])
        width[idx]=score_cp_width(np.array([c]),t,e)[0]
    qhi,_=b.quantile_focused_upper(y,stream,stage)
    return qhi,cost,stage


def width_only_test(qhi,stream,threshold,rho):


    n=len(stream);lo=np.zeros(n);hi=np.ones(n)
    stage=np.zeros(n,dtype=np.int8);active=np.ones(n,dtype=bool)
    weights=np.full(NST,rho/(NST-1));weights[-1]=1-rho
    for j,t in enumerate(ST):
        ix=np.flatnonzero(active)
        if not len(ix):break
        ll,uu=b.cp_interval(stream[ix,t-1],t,b.BETA_TEST*weights[j])
        lo[ix],hi[ix]=b.update_nested(lo[ix],hi[ix],ll,uu)
        stage[ix]=j+1
        active[ix[(hi[ix]-lo[ix])<=threshold]]=False
    sets=np.column_stack([lo<=qhi,1-hi<=qhi])
    return sets,int(np.sum(ST[stage-1])),stage


def prepare_random_holdout(ds,seed):


    from sklearn.datasets import load_breast_cancer,load_wine,load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    if ds=='breast':
        dat=load_breast_cancer();X=dat.data;y=dat.target
    elif ds=='wine01':
        dat=load_wine();m=dat.target<2;X=dat.data[m];y=dat.target[m]
    elif ds=='digits38':
        dat=load_digits();m=np.isin(dat.target,[3,8]);X=dat.data[m];y=(dat.target[m]==8).astype(int)
    else:raise ValueError(ds)
    indices=np.arange(len(y))
    a,rest=train_test_split(indices,test_size=.5,stratify=y,random_state=seed)
    cal,test=train_test_split(rest,test_size=.5,random_state=seed+101,stratify=None)
    scale=StandardScaler().fit(X[a]);xx=scale.transform(X)
    pca=PCA(n_components=4,random_state=seed).fit(xx[a]);zz=pca.transform(xx)
    scale2=StandardScaler().fit(zz[a]);zz=np.pi*.78*np.tanh(scale2.transform(zz)/1.8)
    return (zz[a],y[a]),(zz[cal],y[cal]),(zz[test],y[test])


def init_run(ds,seed,epochs):
    (zt,yt),(zc,yc),(zv,yv)=(prepare_random_holdout(ds,seed) if SPLIT_POLICY=='random_holdout' else b.prepare(ds,seed))
    theta=b.train((zt,yt),seed,epochs)
    pc,pv=b.predict(zc,theta),b.predict(zv,theta)
    rng=np.random.default_rng(4321+seed*1009+len(ds)*113)
    cal=b.sample_prefixes(pc,rng);test=b.sample_prefixes(pv,rng)
    old=b.old_calibration(yc,cal)
    qrank,_=b.quantile_focused_upper(yc,cal,old['stage'])
    rho=calibrated_rho(len(yc))
    rankset,ranktestshots,rankstage=test_with_spending(qrank,test,rho)
    qideal=b.order_threshold(np.where(yc==0,pc,1-pc),old['k'])
    ideal=np.column_stack([pv<=qideal,1-pv<=qideal])
    return dict(ds=ds,seed=seed,yc=yc,yv=yv,cal=cal,test=test,pv=pv,
      qideal=qideal,ideal=ideal,old=old,qrank=qrank,rho=rho,
      rankset=rankset,ranktestshots=ranktestshots,rankstage=rankstage)


def row(run,method,sets,calshots,testshots,qhi,stage_cal=None,stage_test=None,policy=''):
    y=run['yv'];idx=np.arange(len(y));ideal=run['ideal']
    return dict(dataset=run['ds'],seed=run['seed'],method=method,policy=policy,
      n_cal=len(run['yc']),n_test=len(y),cal_shots=int(calshots),test_shots=int(testshots),
      total_shots=int(calshots+testshots),mean_set_size=float(sets.sum(1).mean()),
      coverage=float(sets[idx,y].mean()),
      ideal_set_containment=float(np.all((~ideal)|sets,axis=1).mean()),
      exact_set_match_rank=float(np.all(sets==run['rankset'],axis=1).mean()),
      subset_rank=float(np.all((~sets)|run['rankset'],axis=1).mean()),
      qhi=float(qhi),rank_qhi=float(run['qrank']),
      mean_cal_shots=float(np.mean(ST[stage_cal-1])) if stage_cal is not None else np.nan,
      mean_test_shots=float(np.mean(ST[stage_test-1])) if stage_test is not None else np.nan)


def dev_candidates(run):

    records=[]
    cal_cache={t:width_cal_threshold(run['yc'],run['cal'],t) for t in CAL_THRESHOLDS}
    for ct,(q,calshots,calst) in cal_cache.items():
        for tt in TEST_THRESHOLDS:
            sets,testshots,testst=width_only_test(q,run['test'],tt,run['rho'])
            key=f'width_cal{ct:.2f}_test{tt:.2f}'
            records.append(row(run,key,sets,calshots,testshots,q,calst,testst,policy=f'{ct:.2f}|{tt:.2f}'))
    return records


def select_on_dev(df,rank_df):
    rsz=float(rank_df['mean_set_size'].mean())
    target_coverage=.90
    v=df.groupby('method',as_index=False).agg(shots=('total_shots','mean'),
     size=('mean_set_size','mean'),coverage=('coverage','mean'),
     containment=('ideal_set_containment','mean'))
    v['size_gap_vs_rank']=v['size']-rsz

    ok=v[(v['size']<=rsz+TOL_SET)&(v['coverage']>=target_coverage)&(v['containment']>=.99)]
    if len(ok):
        selected=ok.sort_values(['shots','size','method']).iloc[0]
        status='feasible'
    else:

        v['violation']=np.maximum(0,v['size']-rsz-TOL_SET)
        selected=v.sort_values(['violation','shots','method']).iloc[0]
        status='NOT_MATCHED_ON_DEV'
    return selected['method'],status,v


def quality_stats(d,rank='rank_V3',vs='width_selected'):
    p=d.pivot(index=['dataset','seed'],columns='method',values=['total_shots','mean_set_size','coverage','ideal_set_containment'])
    out={}
    for method in [x for x in d.method.unique() if x!=rank]:
        ds=(p['mean_set_size'][method]-p['mean_set_size'][rank]).to_numpy()
        shot=(p['total_shots'][method]/p['total_shots'][rank]).to_numpy()
        cov=(p['coverage'][method]-p['coverage'][rank]).to_numpy()
        out[method]={
          'ratio_shots_mean':float(np.mean(shot)),
          'mean_paired_shot_saving_pct':float(100*np.mean(1-shot)),
          'mean_paired_set_delta':float(np.mean(ds)),
          'mean_coverage_delta':float(np.mean(cov)),
          'mean_set_nonlarger_fraction':float(np.mean(ds<=1e-12)),
          'mean_set_within_tolerance_fraction':float(np.mean(np.abs(ds)<=TOL_SET+1e-12)),
          'certification_scope':('finite-shot marginal coverage, not ideal-oracle containment' if method=='standard_QCP64' else 'ideal-score containment with beta failure budget'),
        }

        rng=np.random.default_rng(70917)
        groups=[np.where(p.index.get_level_values(0)==dataset)[0] for dataset in DATASETS]
        boot_sh=[];boot_sz=[]
        for _ in range(1600):
            dsi=rng.integers(0,len(groups),size=len(groups))
            ix=np.concatenate([rng.choice(groups[z],size=len(groups[z]),replace=True) for z in dsi])
            boot_sh.append(np.mean(100*(1-shot[ix])));boot_sz.append(np.mean(ds[ix]))
        out[method]['cluster_bootstrap95_shot_saving_pct']=list(np.quantile(boot_sh,[.025,.975]))
        out[method]['cluster_bootstrap95_set_delta']=list(np.quantile(boot_sz,[.025,.975]))
    return out


def plots(hold,dev_grid,output):
    agg=hold.groupby('method').agg(shots=('total_shots','mean'),size=('mean_set_size','mean'),coverage=('coverage','mean'))
    display=['standard_QCP64','rank_V3','V2_endpoint','width_budget_matched','width_test_only','width_selected']
    labels={'rank_V3':'Rank-aware V3','width_selected':'Width-only (quality-matched)',
      'width_budget_matched':'Width-first (budget-matched)',
      'width_test_only':'Width-only test','V2_endpoint':'V2 Endpoint','standard_QCP64':'Standard QCP-64'}
    dy={'standard_QCP64':7,'rank_V3':-22,'V2_endpoint':10,
        'width_budget_matched':7,'width_test_only':-18,'width_selected':8}
    fig,ax=plt.subplots(figsize=(9.5,5.3))
    for name in display:
        t=agg.loc[name];ax.scatter(t.shots/1000,t['size'],s=70)
        ax.annotate(labels[name],(t.shots/1000,t['size']),xytext=(6,dy[name]),textcoords='offset points',fontsize=8.5)
    ax.set(xlabel='Total shots per dataset-seed run (thousands)',ylabel='Mean prediction-set size',
           title='Held-out shot cost and prediction-set quality')
    ax.margins(x=.18,y=.16);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'heldout_shots_vs_size.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.1))
    a=dev_grid[['shots','size']]
    ax.scatter(a.shots/1000,a['size'],s=20,alpha=.30,label='Dev grid (width-only)')
    for name in ['rank_V3','width_selected','width_budget_matched']:
        v=agg.loc[name]
        ax.scatter([v.shots/1000],[v['size']],s=90,label=labels[name])
    ax.set(xlabel='Total shots per run (thousands)',ylabel='Mean prediction-set size',
           title='Development frontier and held-out outcomes')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(output/'quality_frontier.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));order=['rank_V3','V2_endpoint','width_budget_matched','width_test_only','width_selected']
    ax.bar(range(len(order)),agg.loc[order,'shots']/1000)
    ax.set_xticks(range(len(order)));ax.set_xticklabels([labels[m] for m in order],rotation=16,ha='right',fontsize=9)
    ax.set_ylabel('Mean total shots (thousands)');fig.tight_layout();fig.savefig(output/'heldout_shots.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dev-seeds',type=int,default=5)
    parser.add_argument('--eval-seeds',type=int,default=10)
    parser.add_argument('--eval-start',type=int,default=40)
    parser.add_argument('--epochs',type=int,default=130)
    parser.add_argument('--split-policy',choices=['original_stratified','random_holdout'],default='original_stratified')
    parser.add_argument('--output',default=str(ROOT/'RESULTS'))
    args=parser.parse_args();out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    global SPLIT_POLICY
    SPLIT_POLICY=args.split_policy
    assert b.audit_math()=='PASS'
    t0=time.time()
    devrows=[];devrank=[];devbudget=[]
    for ds in DATASETS:
        for seed in range(args.dev_seeds):
            run=init_run(ds,seed,args.epochs)
            devrows+=dev_candidates(run)
            devrank.append(row(run,'rank_V3',run['rankset'],run['old']['shots'],run['ranktestshots'],run['qrank'],run['old']['stage'],run['rankstage']))
            for power in WIDTH_PRIORITY_POWERS:
                q,cs,st=width_cal_budgeted(run['yc'],run['cal'],run['old']['shots'],power)
                s,ts,stt=test_with_spending(q,run['test'],run['rho'])
                devbudget.append(row(run,f'width_priority_{power:.1f}',s,cs,ts,q,st,stt,policy=f'power={power}'))
            print('DEV',ds,seed,flush=True)
    dev=pd.DataFrame(devrows);rankdev=pd.DataFrame(devrank)
    chosen,status,grid=select_on_dev(dev,rankdev)
    budget_dev=pd.DataFrame(devbudget)
    budget_summary=budget_dev.groupby('method',as_index=False).agg(shots=('total_shots','mean'),size=('mean_set_size','mean'),coverage=('coverage','mean'))
    budget_summary.sort_values(['size','shots','method']).to_csv(out/'development_width_budget_variants.csv',index=False)
    budget_choice=budget_summary.sort_values(['size','shots','method']).iloc[0]['method']
    budget_power=float(budget_choice.replace('width_priority_',''))
    grid.to_csv(out/'development_width_grid.csv',index=False)
    pd.concat([dev,rankdev]).to_csv(out/'development_all_runs.csv',index=False)
    calstr,teststr=chosen.replace('width_cal','').split('_test');ct=float(calstr);tt=float(teststr)
    (out/'frozen_selection.json').write_text(json.dumps({
      'chosen':chosen,'cal_width_threshold':ct,'test_width_threshold':tt,
      'budget_selected_priority':budget_choice,'budget_selected_power':budget_power,
      'status':status,'selection_on':'development seeds 0..4 only',
      'split_policy':SPLIT_POLICY,
      'set_tolerance':TOL_SET,'all_certified_same_alpha':b.ALPHA,
      'all_certified_same_beta':b.BETA,
      'heldout_seed_start':args.eval_start,'heldout_seed_count':args.eval_seeds},indent=2))
    print('SELECTED',chosen,status,'params',ct,tt,'budget priority',budget_choice,flush=True)
    rr=[];details=[]
    for ds in DATASETS:
        for seed in range(args.eval_start,args.eval_start+args.eval_seeds):
            run=init_run(ds,seed,args.epochs)
            rankcal=run['old']['shots'];rankset=run['rankset']
            rr.append(row(run,'rank_V3',rankset,rankcal,run['ranktestshots'],run['qrank'],run['old']['stage'],run['rankstage']))

            q,cs,st=width_cal_threshold(run['yc'],run['cal'],ct)
            s,ts,sttest=width_only_test(q,run['test'],tt,run['rho'])
            rr.append(row(run,'width_selected',s,cs,ts,q,st,sttest,policy=chosen))

            q,cs,st=width_cal_budgeted(run['yc'],run['cal'],rankcal)
            s,ts,sttest=test_with_spending(q,run['test'],run['rho'])
            rr.append(row(run,'width_budget_matched',s,cs,ts,q,st,sttest,
                 policy='greedy widest-first; rank-calibration-budget oracle for analysis'))

            q,cs,st=width_cal_budgeted(run['yc'],run['cal'],rankcal,budget_power)
            s,ts,sttest=test_with_spending(q,run['test'],run['rho'])
            rr.append(row(run,'width_budget_tuned',s,cs,ts,q,st,sttest,
                 policy=f'width / next-increment^{budget_power}; matched budget'))

            s,ts,sttest=width_only_test(run['qrank'],run['test'],tt,run['rho'])
            rr.append(row(run,'width_test_only',s,rankcal,ts,run['qrank'],run['old']['stage'],sttest,
                 policy=f'test_width={tt}'))

            sv,tsv,_=b.test_v2(run['qrank'],run['test'],True)
            rr.append(row(run,'V2_endpoint',sv,rankcal,tsv,run['qrank'],run['old']['stage']))
            sq,shotsq,_=b.ordinary_qcp(run['yc'],run['cal'],run['test'],64)
            rr.append(row(run,'standard_QCP64',sq,len(run['yc'])*64,len(run['yv'])*64,np.nan))


            for label,sets,shot in [('rank_V3',rankset,rankcal+run['ranktestshots']),
                                    ('width_selected',width_only_test(width_cal_threshold(run['yc'],run['cal'],ct)[0],run['test'],tt,run['rho'])[0],None)]:
                for i in range(len(run['yv'])):
                    details.append(dict(dataset=ds,seed=seed,index=i,method=label,
                       label_true=int(run['yv'][i]),set_size=int(sets[i].sum()),covered=int(sets[i,run['yv'][i]])))
            print('EVAL',ds,seed,'rank',rr[-7]['total_shots'],'width',rr[-6]['total_shots'],'tuned',rr[-4]['total_shots'],flush=True)
    hold=pd.DataFrame(rr);hold.to_csv(out/'heldout_all_runs.csv',index=False)
    pd.DataFrame(details).to_csv(out/'heldout_sample_details.csv',index=False)
    sm=hold.groupby('method',as_index=False).agg(mean_shots=('total_shots','mean'),sd_shots=('total_shots','std'),
      mean_cal_shots=('cal_shots','mean'),mean_test_shots=('test_shots','mean'),
      mean_set_size=('mean_set_size','mean'),mean_coverage=('coverage','mean'),
      mean_ideal_containment=('ideal_set_containment','mean'),mean_match_rank=('exact_set_match_rank','mean'))
    sm.to_csv(out/'heldout_summary.csv',index=False)
    by=hold.groupby(['dataset','method'],as_index=False).agg(mean_shots=('total_shots','mean'),
      mean_set_size=('mean_set_size','mean'),coverage=('coverage','mean'),containment=('ideal_set_containment','mean'))
    by.to_csv(out/'heldout_by_dataset.csv',index=False)
    pairs=quality_stats(hold)
    (out/'paired_metrics.json').write_text(json.dumps({'selection':status,'chosen':chosen,'budget_selected_power':budget_power,
      'paired':pairs,'split_policy':SPLIT_POLICY,'runtime_seconds':time.time()-t0,'n_heldout_runs':args.eval_seeds*len(DATASETS),
      'n_model_evaluations':len(hold),'sample_n':int(len(details)/2)},indent=2))
    plots(hold,grid,out)
    print('SUMMARY\n',sm.to_string(index=False));print('PAIRED\n',json.dumps(pairs,indent=2))

if __name__=='__main__':main()
