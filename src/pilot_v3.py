import os, sys, math, time, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp

try: import qcp_rank_v2 as base
except ModuleNotFoundError:
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import qcp_rank_v2 as base

OUT=Path(__file__).resolve().parent/'RESULTS'
STAGES=np.array(base.STAGES)
BETA_TEST=base.BETA_TEST

FRACS=np.array([.025,.05,.10,.20,.35,.50,.70,.90])
WEIGHTS=np.array([.07,.10,.14,.20,.19,.14,.10,.06]); assert abs(WEIGHTS.sum()-1)<1e-10


def log_e_mix(count,t,c,direction):


    count=np.asarray(count,dtype=float);c=np.broadcast_to(np.asarray(c,dtype=float),count.shape)
    ans=np.full(count.shape,-np.inf)
    valid=(c>0)&(c<1)
    if not valid.any(): return ans
    cv=c[valid];nv=count[valid]
    if direction=='up': alt=cv[:,None]+(1-cv[:,None])*FRACS[None,:]
    elif direction=='down': alt=cv[:,None]*(1-FRACS[None,:])
    else:raise ValueError(direction)
    log_lrs=nv[:,None]*np.log(alt/cv[:,None])+(t-nv[:,None])*np.log((1-alt)/(1-cv[:,None]))
    ans[valid]=logsumexp(log_lrs+np.log(WEIGHTS),axis=1)


    return ans


def evidence_readout(qhi,stream,cap=1024,split_budget=4,stages=STAGES):


    n=len(stream);alive=np.ones(n,dtype=bool);shots=np.zeros(n,dtype=int)
    result=np.ones((n,2),dtype=bool);status=np.zeros((n,2),dtype=np.int8)
    threshold=np.log(split_budget/BETA_TEST)
    for t in stages:
        if t>cap:break
        ix=np.flatnonzero(alive)
        if not len(ix):break
        c=stream[ix,t-1];b0=np.full(len(ix),float(qhi));b1=1-b0
        e_0_ex=log_e_mix(c,t,b0,'up')>=threshold
        e_0_in=log_e_mix(c,t,b0,'down')>=threshold
        e_1_ex=log_e_mix(c,t,b1,'down')>=threshold
        e_1_in=log_e_mix(c,t,b1,'up')>=threshold
        st0=np.where(e_0_ex,-1,np.where(e_0_in,1,0))
        st1=np.where(e_1_ex,-1,np.where(e_1_in,1,0))
        status[ix,0]=st0;status[ix,1]=st1
        result[ix,0]=st0!=-1;result[ix,1]=st1!=-1
        shots[ix]=t


        resolved=(np.all(status[ix]!=0,axis=1) | ((qhi>=.5) & np.any(status[ix]==-1,axis=1)))
        alive[ix[resolved]]=False
    return result,int(shots.sum()),shots,status


def test_evidence_exclusion(qhi,stream,cap=1024,stages=STAGES):


    n=len(stream);active=np.ones(n,dtype=bool);spent=np.zeros(n,dtype=int)
    sets=np.ones((n,2),dtype=bool);thr=math.log(2/BETA_TEST)
    for t in stages:
        if t>cap:break
        ix=np.flatnonzero(active)
        if len(ix)==0:break
        counts=stream[ix,t-1]
        mask0=log_e_mix(counts,t,np.full(len(ix),qhi),'up')>=thr
        mask1=log_e_mix(counts,t,np.full(len(ix),1-qhi),'down')>=thr
        sets[ix,0]=~mask0;sets[ix,1]=~mask1
        spent[ix]=t
        if qhi>=.5: active[ix[mask0|mask1]]=False
        else: active[ix[mask0&mask1]]=False
    return sets,int(spent.sum()),spent


def evaluate_one(ds,seed,epochs=130):
    (zt,yt),(zc,yc),(zv,yv)=base.prepare(ds,seed)
    theta=base.train((zt,yt),seed,epochs)
    pc,pv=base.predict(zc,theta),base.predict(zv,theta)
    rng=np.random.default_rng(4321+seed*1009+len(ds)*113)
    cal=base.sample_prefixes(pc,rng);test=base.sample_prefixes(pv,rng)
    old=base.old_calibration(yc,cal)
    qhi,_=base.quantile_focused_upper(yc,cal,old['stage'])
    v1,s1,_=base.test_v1(old['qlo'],old['qhi'],test)
    v2,s2,_=base.test_v2(qhi,test,True)
    e4,s4,sper,_=evidence_readout(qhi,test)
    e2,st2,sp2=test_evidence_exclusion(qhi,test)
    fixed,sfix,_=base.fixed_certificate(yc,cal,test,1024)
    qcp,sqcp,_=base.ordinary_qcp(yc,cal,test,64)
    k=old['k']; qtrue=base.order_threshold(np.where(yc==0,pc,1-pc),k)
    ideal=np.column_stack([pv<=qtrue,1-pv<=qtrue])
    methods={
      'V1_adaptive':(v1,old['shots']+s1),
      'V2_endpoint':(v2,old['shots']+s2),
      'V3_eprocess_4tests':(e4,old['shots']+s4),
      'V3_eprocess_2tests':(e2,old['shots']+st2),
      'fixed_certified_1024':(fixed,sfix),
      'ordinary_QCP_64':(qcp,sqcp)
    }
    rows=[]
    for name,(sets,nshots) in methods.items():
        rows.append({'dataset':ds,'seed':seed,'method':name,'n_cal':len(yc),'n_test':len(yv),
          'q_up':qhi,'q_ideal':qtrue,
          'shots_total':nshots,'set_size':float(sets.sum(axis=1).mean()),
          'coverage':float(sets[np.arange(len(yv)),yv].mean()),
          'oracle_containment':float(np.all((~ideal)|sets,axis=1).mean()),
          'test_shots':int(nshots-old['shots']) if name not in ('fixed_certified_1024','ordinary_QCP_64') else np.nan,
          'set_delta_vs_v2':float((sets.sum(axis=1)-v2.sum(axis=1)).mean()),
          'fraction_exact_same_as_v2':float(np.all(sets==v2,axis=1).mean()),
          'cal_shots':old['shots'],'accuracy':float(((pv>=.5)==yv).mean())})

    debug=[]
    for i,p in enumerate(pv):
        debug.append({'dataset':ds,'seed':seed,'test_i':i,'p':float(p),'qhi':qhi,
           'gap_to_exclusion_boundary':float(min(abs(p-qhi),abs(p-(1-qhi)))),
           'e4_shots':int(sper[i]),'e2_shots':int(sp2[i]),
           'v2_set_size':int(v2[i].sum()),'e4_set_size':int(e4[i].sum()),
           'v2_includes_true':bool(v2[i,yv[i]]),'e4_includes_true':bool(e4[i,yv[i]])})
    return rows,debug


def audit_e_process():

    from scipy.stats import binom
    for direction in ['up','down']:
        c=.35
        ps=np.linspace(0,c,8) if direction=='up' else np.linspace(c,1,8)
        for p in ps:


            for t in [8,32,64]:
                ks=np.arange(t+1)
                E=log_e_mix(ks,t,np.full(len(ks),c),direction)
                prob=binom.pmf(ks,t,p)[E>=math.log(1/.05)].sum()
                assert prob<=.05+1e-10,(direction,t,p,prob)
    return 'PASS'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed-start',type=int,default=20)
    parser.add_argument('--seeds',type=int,default=10)
    parser.add_argument('--epochs',type=int,default=130)
    parser.add_argument('--output',default=str(OUT))
    a=parser.parse_args()
    assert audit_e_process()=='PASS'; t0=time.time()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=[];debug=[]
    for ds in ['breast','wine01','digits38']:
        for seed in range(a.seed_start,a.seed_start+a.seeds):
            r,d=evaluate_one(ds,seed,a.epochs);rows+=r;debug+=d
            print(ds,seed,'V2',int(r[1]['shots_total']),'e4',int(r[2]['shots_total']),
                  'e2',int(r[3]['shots_total']), 'size',round(r[1]['set_size'],3),round(r[2]['set_size'],3),flush=True)
    df=pd.DataFrame(rows);dbg=pd.DataFrame(debug)
    df.to_csv(out/'all_runs.csv',index=False);dbg.to_csv(out/'per_test_debug.csv',index=False)
    agg=df.groupby('method',as_index=False).agg(shots=('shots_total','mean'),size=('set_size','mean'),coverage=('coverage','mean'),containment=('oracle_containment','mean'))
    agg.to_csv(out/'summary.csv',index=False)
    piv=df.pivot(index=['dataset','seed'],columns='method',values=['shots_total','set_size'])
    res={'seed_start':a.seed_start,'seeds':a.seeds,'n_runs':len(df),'n_test_samples':len(dbg),
      'audit_eprocess':audit_e_process(),
      'savings_e4_vs_v2_pct':float((100*(1-piv['shots_total']['V3_eprocess_4tests']/piv['shots_total']['V2_endpoint'])).mean()),
      'savings_e2_vs_v2_pct':float((100*(1-piv['shots_total']['V3_eprocess_2tests']/piv['shots_total']['V2_endpoint'])).mean()),
      'size_delta_e4_vs_v2':float((piv['set_size']['V3_eprocess_4tests']-piv['set_size']['V2_endpoint']).mean()),
      'size_delta_e2_vs_v2':float((piv['set_size']['V3_eprocess_2tests']-piv['set_size']['V2_endpoint']).mean()),
      'elapsed_seconds':time.time()-t0}
    (out/'metrics.json').write_text(json.dumps(res,indent=2))
    print(agg.to_string(index=False));print(json.dumps(res,indent=2))

if __name__=='__main__':main()
