import os, math, sys, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import binom, norm
import qcp_rank_v2 as base
from pilot_v3 import evidence_readout, audit_e_process

STAGES=base.STAGES
OUT=Path(__file__).resolve().parent/'RESULTS'

def calibrated_rho(n):


    return max(.2,min(.7,32./n))

def test_with_spending(qhi,streams,rho):
    n=len(streams);lo=np.zeros(n);hi=np.ones(n)
    active=np.ones(n,dtype=bool);stage=np.zeros(n,dtype=int)
    weights=np.full(len(STAGES),rho/(len(STAGES)-1));weights[-1]=1-rho
    for j,t in enumerate(STAGES):
        ix=np.flatnonzero(active)
        if not len(ix):break
        ll,uu=base.cp_interval(streams[ix,t-1],t,base.BETA_TEST*weights[j])
        lo[ix],hi[ix]=base.update_nested(lo[ix],hi[ix],ll,uu)
        stage[ix]=j+1
        scorelo=np.column_stack([lo[ix],1-hi[ix]])
        scorehi=np.column_stack([hi[ix],1-lo[ix]])
        active[ix[np.all((scorehi<=qhi)|(scorelo>qhi),axis=1)]]=False
    sets=np.column_stack([lo<=qhi,1-hi<=qhi])
    return sets,int(np.sum(np.asarray(STAGES)[stage-1])),stage

def ranktail_gamma(n,m,eta=base.BETA_CAL):
    if m<0 or m>=n:raise ValueError((n,m))
    return brentq(lambda g:binom.sf(m,n,g)-eta,1e-12,.5,xtol=1e-14)

def ranktail_quantile(y,calstream,t=256):
    n=len(y);k=math.ceil((n+1)*(1-base.ALPHA+base.BETA))
    if k>n:return 1.,0,0.
    m=max(0,min(n-k,max(1,int(.025*n))))
    gamma=ranktail_gamma(n,m)
    raw=calstream[:,t-1]
    succ=np.where(y==0,raw,t-raw)
    uppers=base.one_sided_upper(succ,t,gamma)
    return base.order_threshold(uppers,k+m),m,gamma

METHODS=[
 'V1_adaptive','V2_endpoint','V3_size_adaptive','V3_rho_020','V3_eprocess_4tests',
 'V3_ranktail_cal_256','fixed_certified_256','fixed_certified_512',
 'fixed_certified_1024','standard_QCP_64','oracle_ideal']

def one_run(ds,seed,epochs=130):
    (zt,yt),(zc,yc),(zv,yv)=base.prepare(ds,seed)
    th=base.train((zt,yt),seed,epochs)
    pc,pv=base.predict(zc,th),base.predict(zv,th)
    rng=np.random.default_rng(4321+seed*1009+len(ds)*113)
    cal=base.sample_prefixes(pc,rng);test=base.sample_prefixes(pv,rng)
    old=base.old_calibration(yc,cal);qhi,_=base.quantile_focused_upper(yc,cal,old['stage'])
    qtrue=base.order_threshold(np.where(yc==0,pc,1-pc),old['k'])
    ideal=np.column_stack([pv<=qtrue,1-pv<=qtrue])
    v1,c1,_=base.test_v1(old['qlo'],old['qhi'],test)
    v2,c2,_=base.test_v2(qhi,test,True)
    rho=calibrated_rho(len(yc));v3,c3,s3=test_with_spending(qhi,test,rho)
    v3_fix,ccfix,sfix=test_with_spending(qhi,test,.20)
    ve,ce,se,_=evidence_readout(qhi,test)
    qtail,m,gamma=ranktail_quantile(yc,cal,256)
    vt,ct,_=base.test_v2(qtail,test,True)


    fixed={t:base.fixed_certificate(yc,cal,test,t) for t in (256,512,1024)}
    qcp,cqcp,_=base.ordinary_qcp(yc,cal,test,64)
    methods={
       'V1_adaptive':(v1,old['shots']+c1),
       'V2_endpoint':(v2,old['shots']+c2),
       'V3_size_adaptive':(v3,old['shots']+c3),
       'V3_rho_020':(v3_fix,old['shots']+ccfix),
       'V3_eprocess_4tests':(ve,old['shots']+ce),
       'V3_ranktail_cal_256':(vt,len(yc)*256+ct),
       'fixed_certified_256':(fixed[256][0],fixed[256][1]),
       'fixed_certified_512':(fixed[512][0],fixed[512][1]),
       'fixed_certified_1024':(fixed[1024][0],fixed[1024][1]),
       'standard_QCP_64':(qcp,cqcp),
       'oracle_ideal':(ideal,np.nan),
    }
    rows=[];details=[]
    for method in METHODS:
        sets,shots=methods[method]
        rows.append(dict(dataset=ds,seed=seed,method=method,n_train=len(yt),n_cal=len(yc),n_test=len(yv),
            rho=rho,qhi=qhi,qtrue=qtrue,ranktail_m=m,ranktail_gamma=gamma,
            total_shots=shots,mean_set_size=float(sets.sum(axis=1).mean()),
            coverage=float(sets[np.arange(len(yv)),yv].mean()),
            oracle_containment=float(np.all((~ideal)|sets,axis=1).mean()),
            set_equal_v2=float(np.all(sets==v2,axis=1).mean()),
            set_size_delta_v2=float((sets.sum(axis=1)-v2.sum(axis=1)).mean()),
            v1_cal_shots=int(old['shots']),
            classifier_accuracy=float(((pv>=.5)==yv).mean()),
            certification_target_beta=base.BETA))
    for i,p in enumerate(pv):
        details.append(dict(dataset=ds,seed=seed,index=i,prob=float(p),
          qhi=float(qhi),margin=float(min(abs(p-qhi),abs(p-(1-qhi)))),
          V3_test_shots=int(np.asarray(STAGES)[s3[i]-1]),
          eprocess_test_shots=int(se[i]),
          V3_set_size=int(v3[i].sum()),V2_set_size=int(v2[i].sum()),
          truth=int(yv[i]),V3_covered=bool(v3[i,yv[i]])))
    return rows,details

def kl_bern(p,q):
    p=np.asarray(p,dtype=float);q=np.asarray(q,dtype=float)
    return p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))

def kl_decision_lower_bound(c,g,err):


    if not (0<err<.5 and 0<g<min(c,1-c)):return float('nan')
    p0=c-g;p1=c+g
    d=(1-err)*np.log((1-err)/err)+err*np.log(err/(1-err))
    return float(d/kl_bern(p1,p0))

def audit_quantile():

    for n,k in [(142,131),(89,82),(32,31)]:
        m=min(n-k,max(1,int(.025*n)));g=ranktail_gamma(n,m)
        assert abs(binom.sf(m,n,g)-base.BETA_CAL)<1e-9
        rng=np.random.default_rng(920+n)
        for _ in range(250):
            actual=rng.uniform(size=n)
            u=np.minimum(1,actual+rng.uniform(0,.5,n))
            bad=rng.choice(n,size=m,replace=False)
            u[bad]=np.maximum(0,actual[bad]-.15)
            assert base.order_threshold(actual,k) <= base.order_threshold(u,k+m)+1e-12
    assert audit_e_process()=='PASS'

    for g in [.025,.05,.1,.2]:
        assert kl_decision_lower_bound(.5,g,.01)>0
    return 'PASS'

def plot_results(df,debug,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gp=df.groupby('method').agg(shots=('total_shots','mean'),size=('mean_set_size','mean'),coverage=('coverage','mean'))
    order=[m for m in METHODS if m!='oracle_ideal']
    fig,ax=plt.subplots(figsize=(9,5))
    for m in order:
        r=gp.loc[m];ax.scatter(r.shots/1000,r['size'],s=48)
        ax.annotate(m,(r.shots/1000,r['size']),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Mean total shots per dataset-seed run (thousands)');ax.set_ylabel('Mean prediction-set size')
    ax.set_title('Coverage certificate: shot cost vs set size');ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/'shot_vs_size.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    keep=['V2_endpoint','V3_size_adaptive','V3_eprocess_4tests','V3_ranktail_cal_256','fixed_certified_1024']
    axs=np.arange(len(keep));ax.bar(axs,[gp.loc[m,'shots']/1000 for m in keep]);ax.set_xticks(axs)
    ax.set_xticklabels(keep,rotation=20,ha='right',fontsize=9);ax.set_ylabel('Mean total shots (thousands)')
    fig.tight_layout();fig.savefig(out/'shots_bar.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    x=np.maximum(debug['margin'].to_numpy(),1e-4)
    y=debug['V3_test_shots'].to_numpy()
    ax.scatter(x,y,alpha=.35,s=10)
    ax.set_xscale('log');ax.set_xlabel('Gap to nearest conformal decision boundary')
    ax.set_ylabel('V3 test shots (predeclared stages)');ax.set_title('Boundary difficulty and sample cost')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'margin_vs_shots.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));c=.5;err=.01
    gs=np.geomspace(.005,.2,80)
    ax.plot(gs,[kl_decision_lower_bound(c,g,err) for g in gs])
    ax.set_xscale('log');ax.set_yscale('log')
    ax.set_xlabel('Separation g (Bernoulli probabilities 0.5-g vs 0.5+g)')
    ax.set_ylabel('Information-theoretic lower bound on E[T]')
    ax.set_title('Reference: two-sided fixed-confidence threshold decision')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'kl_lower_bound.png',dpi=180);plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=10)
    p.add_argument('--seed-start',type=int,default=30)
    p.add_argument('--epochs',type=int,default=130)
    p.add_argument('--output',default=str(OUT))
    args=p.parse_args();out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    assert base.audit_math()=='PASS';assert audit_quantile()=='PASS'
    time0=time.time();rr=[];dd=[]
    for ds in ['breast','wine01','digits38']:
        for seed in range(args.seed_start,args.seed_start+args.seeds):
            r,d=one_run(ds,seed,args.epochs);rr.extend(r);dd.extend(d)
            v2=r[METHODS.index('V2_endpoint')];v3=r[METHODS.index('V3_size_adaptive')]
            print(f'{ds:8s} seed {seed}: V2 {v2["total_shots"]:6.0f} -> V3 {v3["total_shots"]:6.0f} '
                  f'set {v2["mean_set_size"]:.3f} -> {v3["mean_set_size"]:.3f}',flush=True)
    df=pd.DataFrame(rr);debug=pd.DataFrame(dd)
    df.to_csv(out/'all_methods_runs.csv',index=False)
    debug.to_csv(out/'test_sample_details.csv',index=False)
    sm=df.groupby('method',as_index=False).agg(mean_shots=('total_shots','mean'),sd_shots=('total_shots','std'),
        mean_set=('mean_set_size','mean'),mean_coverage=('coverage','mean'),oracle_containment=('oracle_containment','mean'))
    sm.to_csv(out/'summary.csv',index=False)
    byds=df.groupby(['dataset','method'],as_index=False).agg(mean_shots=('total_shots','mean'),mean_set=('mean_set_size','mean'),mean_coverage=('coverage','mean'))
    byds.to_csv(out/'by_dataset.csv',index=False)
    pivot=df.pivot(index=['dataset','seed'],columns='method',values=['total_shots','mean_set_size'])
    v2s=pivot['total_shots']['V2_endpoint']
    v3s=pivot['total_shots']['V3_size_adaptive']
    size2=pivot['mean_set_size']['V2_endpoint'];size3=pivot['mean_set_size']['V3_size_adaptive']
    save=(1-v3s/v2s).to_numpy();dsize=(size3-size2).to_numpy()
    rng=np.random.default_rng(121);indices=rng.integers(0,len(save),size=(10000,len(save)))
    boot=lambda arr:np.quantile(arr[indices].mean(axis=1),[.025,.975]).tolist()
    metric={'date':'2026-11-13','n_datasets':3,'seeds':args.seeds,'start_seed':args.seed_start,
      'n_fit_runs':args.seeds*3,'n_method_evaluations':len(df),'n_test_points':len(debug),
      'audit': 'PASS', 'alpha':base.ALPHA,'beta':base.BETA,
      'mean_saving_vs_V2_fraction':float(save.mean()),'bootstrap_saving_fraction_95':boot(save),
      'mean_set_delta_vs_V2':float(dsize.mean()),'bootstrap_set_delta_95':boot(dsize),
      'same_or_smaller_set_runs':int(np.sum(dsize<=1e-12)),
      'all_runs':len(dsize), 'mean_containment_V3':float(df[df.method=='V3_size_adaptive'].oracle_containment.mean()),
      'runtime_seconds':float(time.time()-time0),
      'caveat':'Bootstrap is descriptive over dataset-seed blocks; only three related source datasets.',
      'shot_count':'Hypothetical physical executions of measured prefixes; full counterfactual streams pre-generated for pairing.',
      'coverage_notice':'Per-query marginal coverage; standard QCP64 is a different finite-shot guarantee and not ideal-oracle containment.'}
    (out/'metrics.json').write_text(json.dumps(metric,indent=2))
    plot_results(df,debug,out)
    print('\nFINAL RESULTS\n',sm.to_string(index=False))
    print('\nMETRICS\n',json.dumps(metric,indent=2))

if __name__=='__main__':main()
