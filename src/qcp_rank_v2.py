import argparse, math, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

OUT = Path(__file__).resolve().parent
STAGES = [8,16,32,64,128,256,512,1024]
ALPHA, BETA = 0.10, 0.01
BETA_CAL, BETA_TEST = 0.005, 0.005


GATE_PAIRS = {}
CX_PERMS = {}
for j in range(4):
    step = 1 << (3-j)
    GATE_PAIRS[j] = (np.array([i for i in range(16) if (i & step) == 0]),
                     np.array([i for i in range(16) if (i & step) == 0])+step)
for c,t in [(0,1),(1,2),(2,3),(3,0),(1,0),(2,0)]:
    CX_PERMS[(c,t)] = torch.tensor([i ^ (1 << (3-t)) if (i & (1 << (3-c))) else i for i in range(16)],dtype=torch.long)
MEASURE1 = torch.tensor([i for i in range(16) if i & 8],dtype=torch.long)

def ry(state, angle, wire):
    idx0,idx1 = GATE_PAIRS[wire]
    a,b = state[:,idx0],state[:,idx1]
    cs = torch.cos(angle/2).reshape(-1,1); sn = torch.sin(angle/2).reshape(-1,1)
    out = state.clone()
    out[:,idx0] = cs*a - sn*b
    out[:,idx1] = sn*a + cs*b
    return out

def model_p(x, theta):
    n = x.shape[0]
    s = torch.zeros((n,16),dtype=x.dtype,device=x.device)
    s[:,0]=1.0
    for j in range(4): s=ry(s,x[:,j],j)
    for c,t in [(0,1),(1,2),(2,3)]: s=s[:,CX_PERMS[(c,t)]]
    for j in range(4): s=ry(s,theta[j].expand(n)+0.45*x[:,j],j)
    for c,t in [(3,0),(2,0),(1,0)]: s=s[:,CX_PERMS[(c,t)]]
    for j in range(4): s=ry(s,theta[j+4].expand(n),j)
    return torch.sum(s[:,MEASURE1]**2,axis=1).clamp(1e-6,1-1e-6)

def prepare(name, seed):
    if name=='breast':
        data=load_breast_cancer(); X=data.data; y=data.target
    elif name=='wine01':
        data=load_wine(); m=data.target<2; X=data.data[m]; y=data.target[m]
    elif name=='digits38':
        data=load_digits(); m=np.isin(data.target,[3,8]); X=data.data[m]; y=(data.target[m]==8).astype(int)
    else: raise ValueError(name)
    ind=np.arange(len(y))
    a,b=train_test_split(ind,test_size=0.5,stratify=y,random_state=seed)
    c,d=train_test_split(b,test_size=0.5,stratify=y[b],random_state=seed+101)
    scaler=StandardScaler().fit(X[a]); xs=scaler.transform(X)
    pca=PCA(n_components=4,random_state=seed).fit(xs[a]); z=pca.transform(xs)
    sc=StandardScaler().fit(z[a]); z=np.pi*0.78*np.tanh(sc.transform(z)/1.8)
    return (z[a],y[a]),(z[c],y[c]),(z[d],y[d])

def train(train_xy, seed, epochs):
    torch.manual_seed(seed)
    X,y=train_xy
    xt=torch.as_tensor(X,dtype=torch.float64)
    yt=torch.as_tensor(y,dtype=torch.float64)
    theta=torch.nn.Parameter(torch.randn(8,dtype=torch.float64)*0.25)
    opt=torch.optim.Adam([theta],lr=0.07)
    bestloss=float('inf'); best=None
    for k in range(epochs):
        opt.zero_grad(); p=model_p(xt,theta)
        loss=-(yt*torch.log(p)+(1-yt)*torch.log(1-p)).mean()+0.0001*(theta**2).mean()
        loss.backward(); opt.step()
        if loss.item()<bestloss: bestloss=loss.item(); best=theta.detach().clone()
    return best

def predict(z, theta):
    with torch.no_grad(): return model_p(torch.tensor(z,dtype=torch.float64),theta).numpy()

def score_interval(p_lo,p_hi,y):
    return np.where(y==1,1-p_hi,p_lo),np.where(y==1,1-p_lo,p_hi)

def cp_interval(successes,t,err):
    successes=np.asarray(successes,dtype=float)
    lo=np.where(successes>0,beta.ppf(err/2, successes,np.maximum(t-successes+1,1)),0.)
    hi=np.where(successes<t,beta.ppf(1-err/2,successes+1,np.maximum(t-successes,1)),1.)
    return np.maximum(0,np.nan_to_num(lo,nan=0)),np.minimum(1,np.nan_to_num(hi,nan=1))

def order_threshold(v,k):
    return 1.0 if k>len(v) else float(np.partition(v,k-1)[k-1])

def calibrate(p,y,rng,alpha=ALPHA):
    n=len(y); k=math.ceil((n+1)*(1-alpha+BETA))
    p_lo=np.zeros(n); p_hi=np.ones(n); successes=np.zeros(n,dtype=int)
    stage=np.zeros(n,dtype=int); stopped=np.zeros(n,dtype=bool)
    e=BETA_CAL/(n*len(STAGES))
    qlo,qhi=0.,1.
    for stage_idx,t in enumerate(STAGES):
        upd=np.where(~stopped)[0]
        if len(upd)==0: break
        delta=t-np.where(stage[upd]>0,np.array(STAGES)[stage[upd]-1],0)
        successes[upd]+=rng.binomial(delta,p[upd]); stage[upd]=stage_idx+1
        l,u=cp_interval(successes[upd],t,e)
        p_lo[upd]=np.maximum(p_lo[upd],l);p_hi[upd]=np.minimum(p_hi[upd],u)
        s_lo,s_hi=score_interval(p_lo,p_hi,y)
        qlo=order_threshold(s_lo,k);qhi=order_threshold(s_hi,k)


        stopped = ((s_hi < qlo) | (s_lo > qhi))

    s_lo,s_hi=score_interval(p_lo,p_hi,y)
    qlo,qhi=order_threshold(s_lo,k),order_threshold(s_hi,k)
    return qlo,qhi,int(sum(np.array(STAGES)[np.maximum(0,stage-1)])),(int((~stopped).sum())),k

def certify_test(p, qlo,qhi,rng):
    cnt=0; prev=0; plo=0.;phi=1.;nshots=STAGES[-1]
    e=BETA_TEST/len(STAGES)
    for t in STAGES:
        cnt+=rng.binomial(t-prev,p);prev=t
        l,u=cp_interval(cnt,t,e)
        plo=max(plo,float(l));phi=min(phi,float(u))

        a_lo=np.array([plo,1-phi]);a_hi=np.array([phi,1-plo])
        status=np.where(a_hi<=qlo,1,np.where(a_lo>qhi,-1,0))
        if np.all(status!=0): nshots=t;break

    set_y=np.array([int(plo<=qhi),int(1-phi<=qhi)],dtype=int)
    resolved=int(np.all(status!=0))
    return set_y,nshots,resolved

def pset_from_scores(prob,q):
    return np.column_stack([prob<=q,1-prob<=q])


CAL_EARLY_FRAC = 0.45
TEST_EARLY_FRAC = 0.05
METHODS = ['V1_adaptive','V2_rank_upper','V2_endpoint','fixed_certified_256',
           'fixed_certified_512','fixed_certified_1024','standard_QCP_64','oracle_ideal']


def sample_prefixes(probs, rng):


    probs=np.asarray(probs)
    return np.cumsum((rng.random((len(probs),1024)) < probs[:,None]).astype(np.int16),axis=1)


def one_sided_upper(count, t, err):

    c=np.asarray(count)
    return np.where(c<t,beta.ppf(1-err,c+1,np.maximum(t-c,1)),1.)


def update_nested(lo_prev, hi_prev, candidate_lo, candidate_hi):


    lo_next=np.maximum(lo_prev,candidate_lo)
    hi_next=np.minimum(hi_prev,candidate_hi)
    conflict=lo_next>hi_next
    lo_next=np.where(conflict,lo_prev,lo_next)
    hi_next=np.where(conflict,hi_prev,hi_next)
    return lo_next,hi_next

def old_calibration(y, shots):

    n=len(y); k=math.ceil((n+1)*(1-ALPHA+BETA))
    pl=np.zeros(n);pu=np.ones(n)
    stage=np.zeros(n,dtype=int);active=np.ones(n,dtype=bool)
    err=BETA_CAL/(n*len(STAGES))
    for j,t in enumerate(STAGES):
        idx=np.flatnonzero(active)
        if not len(idx):break
        lo,hi=cp_interval(shots[idx,t-1],t,err)
        pl[idx],pu[idx]=update_nested(pl[idx],pu[idx],lo,hi)
        stage[idx]=j+1
        s_l,s_u=score_interval(pl,pu,y)
        qlo=order_threshold(s_l,k);qhi=order_threshold(s_u,k)
        active=~((s_u<qlo)|(s_l>qhi))
    s_l,s_u=score_interval(pl,pu,y)
    return dict(qlo=order_threshold(s_l,k),qhi=order_threshold(s_u,k),
        stage=stage, shots=int(np.sum(np.array(STAGES)[stage-1])),k=k)


def quantile_focused_upper(y,shots,stages,earlyfrac=CAL_EARLY_FRAC):


    n=len(y);k=math.ceil((n+1)*(1-ALPHA+BETA))
    nstages=len(STAGES); upper=np.ones(n)
    for j,t in enumerate(STAGES):
        ix=np.flatnonzero(stages>=j+1)
        if not len(ix):continue
        c=shots[ix,t-1]; score_success=np.where(y[ix]==0,c,t-c)
        err=BETA_CAL/n*(earlyfrac/(nstages-1) if j<nstages-1 else 1-earlyfrac)
        upper[ix]=np.minimum(upper[ix],one_sided_upper(score_success,t,err))
    return order_threshold(upper,k),upper


def test_v1(qlo,qhi,stream):
    n=len(stream);lo=np.zeros(n);hi=np.ones(n)
    stages=np.zeros(n,dtype=int);active=np.ones(n,dtype=bool)
    for j,t in enumerate(STAGES):
        ix=np.flatnonzero(active)
        if not len(ix):break
        ll,uu=cp_interval(stream[ix,t-1],t,BETA_TEST/len(STAGES))
        lo[ix],hi[ix]=update_nested(lo[ix],hi[ix],ll,uu)
        stages[ix]=j+1
        sl=np.column_stack([lo[ix],1-hi[ix]])
        su=np.column_stack([hi[ix],1-lo[ix]])
        active[ix[np.all((su<=qlo)|(sl>qhi),axis=1)]]=False
    sets=np.column_stack([lo<=qhi,1-hi<=qhi])
    return sets,int(np.sum(np.array(STAGES)[stages-1])),stages


def test_v2(qhi,stream,endpoint=False):


    n=len(stream);lo=np.zeros(n);hi=np.ones(n)
    stage=np.zeros(n,dtype=int);active=np.ones(n,dtype=bool)
    if endpoint:
        weights=np.repeat(TEST_EARLY_FRAC/(len(STAGES)-1),len(STAGES))
        weights[-1]=1-TEST_EARLY_FRAC
    else:weights=np.ones(len(STAGES))/len(STAGES)
    for j,t in enumerate(STAGES):
        ix=np.flatnonzero(active)
        if not len(ix):break
        ll,uu=cp_interval(stream[ix,t-1],t,BETA_TEST*weights[j])
        lo[ix],hi[ix]=update_nested(lo[ix],hi[ix],ll,uu)
        stage[ix]=j+1
        sl=np.column_stack([lo[ix],1-hi[ix]])
        su=np.column_stack([hi[ix],1-lo[ix]])
        active[ix[np.all((su<=qhi)|(sl>qhi),axis=1)]]=False
    sets=np.column_stack([lo<=qhi,1-hi<=qhi])
    return sets,int(np.sum(np.array(STAGES)[stage-1])),stage


def fixed_certificate(y,cal_stream,test_stream,t):
    n=len(y);k=math.ceil((n+1)*(1-ALPHA+BETA))
    pcl,pch=cp_interval(cal_stream[:,t-1],t,BETA_CAL/n)
    _,su=score_interval(pcl,pch,y)
    qhi=order_threshold(su,k)
    ptl,pth=cp_interval(test_stream[:,t-1],t,BETA_TEST)
    sets=np.column_stack([ptl<=qhi,1-pth<=qhi])
    return sets,int((n+len(test_stream))*t),qhi


def ordinary_qcp(y,cal_stream,test_stream,t=64):
    n=len(y);k=math.ceil((n+1)*(1-ALPHA))
    pc=cal_stream[:,t-1]/t
    qs=order_threshold(np.where(y==0,pc,1-pc),k)
    pt=test_stream[:,t-1]/t
    return np.column_stack([pt<=qs,1-pt<=qs]),int((n+len(test_stream))*t),qs


def evaluate_one(dataset,seed,epochs=130):
    (zt,yt),(zc,yc),(zv,yv)=prepare(dataset,seed)
    theta=train((zt,yt),seed,epochs)
    pc,pv=predict(zc,theta),predict(zv,theta)

    rng=np.random.default_rng(4321+seed*1009+len(dataset)*113)
    sc=sample_prefixes(pc,rng);sv=sample_prefixes(pv,rng)
    old=old_calibration(yc,sc)
    newq,_=quantile_focused_upper(yc,sc,old['stage'])
    if newq>old['qhi']+1e-10:
        raise AssertionError('V2 upper quantile must be <= V1 on shared outcomes')
    oracleq=order_threshold(np.where(yc==0,pc,1-pc),old['k'])
    oracle=np.column_stack([pv<=oracleq,1-pv<=oracleq])
    v1,s1,g1=test_v1(old['qlo'],old['qhi'],sv)
    v2a,s2a,g2a=test_v2(newq,sv,False)
    v2b,s2b,g2b=test_v2(newq,sv,True)
    if not np.all((~v2a)|v1):
        raise AssertionError('Pathwise subset property V2_rank_upper subset V1 violated')
    methods={
      'V1_adaptive':(v1,old['shots']+s1),
      'V2_rank_upper':(v2a,old['shots']+s2a),
      'V2_endpoint':(v2b,old['shots']+s2b),
      'oracle_ideal':(oracle,np.nan)
    }
    qfix={}
    for t in [256,512,1024]:
        s,c,q=fixed_certificate(yc,sc,sv,t)
        methods[f'fixed_certified_{t}']=(s,c);qfix[t]=q
    s,c,q=ordinary_qcp(yc,sc,sv)
    methods['standard_QCP_64']=(s,c)
    endpoint_match_fraction=float(np.mean(np.all(v2b==methods['fixed_certified_1024'][0],axis=1)))

    rows=[];ti=np.arange(len(yv))
    for name in METHODS:
        s,spent=methods[name]
        rows.append(dict(dataset=dataset,seed=seed,method=name,n_train=len(yt),n_cal=len(yc),n_test=len(yv),
          mean_set_size=float(np.mean(s.sum(axis=1))),coverage=float(np.mean(s[ti,yv])),
          oracle_containment=float(np.mean(np.all((~oracle)|s,axis=1))),
          total_shots=spent,mean_test_shots=np.nan if name=='oracle_ideal' else (
            float({'V1_adaptive':s1,'V2_rank_upper':s2a,'V2_endpoint':s2b}.get(name,0))/len(yv) if name in ('V1_adaptive','V2_rank_upper','V2_endpoint') else
            float(name.split('_')[-1]) if name.startswith('fixed_') else 64.0),
          test_accuracy=float(np.mean((pv>=.5)==yv)),
          alpha=ALPHA,beta=BETA,q_oracle=oracleq,q_v1=old['qhi'],q_v2=newq,q_fixed_1024=qfix[1024],
          v1_cal_shots=old['shots'],v2_subset_v1=bool(np.all((~v2a)|v1)),
          endpoint_fixed1024_match_fraction=endpoint_match_fraction))
    return rows


def paired_summary(df):
    gg=df.groupby('method',sort=False)
    agg=gg.agg(mean_set=('mean_set_size','mean'),sd_set=('mean_set_size','std'),
               mean_coverage=('coverage','mean'),mean_containment=('oracle_containment','mean'),
               mean_shots=('total_shots','mean')).reset_index()
    pivot=df.pivot_table(index=['dataset','seed'],columns='method',values=['total_shots','mean_set_size'])
    baseline=pivot[('total_shots','V1_adaptive')]
    fixed=pivot[('total_shots','fixed_certified_1024')]
    for i,row in agg.iterrows():
        m=row['method']
        if m!='oracle_ideal':
            agg.loc[i,'shots_saved_vs_v1_pct']=100*(1-pivot[('total_shots',m)].div(baseline)).mean()
            agg.loc[i,'shots_saved_vs_fixed1024_pct']=100*(1-pivot[('total_shots',m)].div(fixed)).mean()
            agg.loc[i,'size_delta_vs_fixed1024']=float((pivot[('mean_set_size',m)]-pivot[('mean_set_size','fixed_certified_1024')]).mean())
    return agg


def audit_math():


    from scipy.stats import binom
    for t in [8,16,32,64]:
      for p in [0.02,0.10,0.3,0.5,0.8,0.98]:
        c=np.arange(t+1)
        up=one_sided_upper(c,t,0.025)
        under=binom.pmf(c,t,p)[up<p-1e-12].sum()
        if under>0.025+1e-11:raise AssertionError((t,p,under))

    from numpy.linalg import norm
    theta=.73
    state=torch.zeros((1,16),dtype=torch.float64);state[:,0]=1
    out=ry(state,torch.tensor([theta],dtype=torch.float64),0).numpy().ravel()
    expected=np.zeros(16);expected[0]=math.cos(theta/2);expected[8]=math.sin(theta/2)
    assert norm(out-expected)<1e-12

    eta_old=BETA_CAL/(2*100*len(STAGES))
    for j in range(len(STAGES)):
        enew=BETA_CAL/100*(CAL_EARLY_FRAC/(len(STAGES)-1) if j<len(STAGES)-1 else 1-CAL_EARLY_FRAC)
        assert enew>=eta_old
    assert abs(TEST_EARLY_FRAC+ (1-TEST_EARLY_FRAC)-1.)<1e-12
    return 'PASS'


def build_figures(df,outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=paired_summary(df)
    sub=p[p['method']!='oracle_ideal']
    order=['V1_adaptive','V2_rank_upper','V2_endpoint','fixed_certified_256',
           'fixed_certified_512','fixed_certified_1024','standard_QCP_64']
    vv=sub.set_index('method').loc[order]
    plt.figure(figsize=(9,5))
    for i,(x,y) in enumerate(zip(vv['mean_shots'],vv['mean_set'])):
        plt.scatter(x/1000,y,s=48)
        plt.annotate(order[i],(x/1000,y),xytext=(4,5),textcoords='offset points',fontsize=8)
    plt.xlabel('Mean total shots per run (thousands)')
    plt.ylabel('Mean prediction-set size')
    plt.title('Certified shot-cost versus set-size trade-off')
    plt.grid(alpha=.25);plt.tight_layout();plt.savefig(outdir/'shot_cost_vs_set_size.png',dpi=180);plt.close()
    selected=['V1_adaptive','V2_rank_upper','V2_endpoint','fixed_certified_1024','standard_QCP_64']
    vals=sub.set_index('method').loc[selected]
    plt.figure(figsize=(9,4));plt.bar(range(len(selected)),vals['mean_set'])
    plt.xticks(range(len(selected)),selected,rotation=18,ha='right')
    plt.ylabel('Mean prediction-set size');plt.title('Conformal set sizes (held-out seeds)')
    plt.tight_layout();plt.savefig(outdir/'set_size.png',dpi=180);plt.close()
    plt.figure(figsize=(9,4));plt.bar(range(len(selected)),vals['mean_coverage'])
    plt.axhline(.90,color='black',ls='--',label='Nominal 90%');plt.legend()
    plt.xticks(range(len(selected)),selected,rotation=18,ha='right');plt.ylim(.75,1.02)
    plt.ylabel('Empirical test coverage');plt.tight_layout();plt.savefig(outdir/'coverage.png',dpi=180);plt.close()


def main():
    parser=argparse.ArgumentParser(description='Quantum CP certified rank-aware V2')
    parser.add_argument('--seeds',type=int,default=10,help='Number of held-out seeds')
    parser.add_argument('--seed-start',type=int,default=10,help='Evaluation seeds start at 10 (dev 0..4)')
    parser.add_argument('--epochs',type=int,default=130)
    parser.add_argument('--output',type=str,default=str(OUT/'RESULTS'))
    args=parser.parse_args()
    t0=time.time();assert audit_math()=='PASS'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for ds in ['breast','wine01','digits38']:
      for seed in range(args.seed_start,args.seed_start+args.seeds):
        item=evaluate_one(ds,seed,epochs=args.epochs)
        rows.extend(item)
        v1=item[0];v2=item[2]
        print(ds,seed,'size',round(v1['mean_set_size'],3),'->',round(v2['mean_set_size'],3),
             'shots',int(v1['total_shots']),'->',int(v2['total_shots']),flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'all_method_runs.csv',index=False)
    sm=paired_summary(df);sm.to_csv(out/'summary_by_method.csv',index=False)
    by_ds=df.groupby(['dataset','method'],as_index=False).agg(coverage=('coverage','mean'),
                set_size=('mean_set_size','mean'),shots=('total_shots','mean'),
                containment=('oracle_containment','mean'))
    by_ds.to_csv(out/'summary_by_dataset.csv',index=False)
    build_figures(df,out)
    pairs=df.pivot(index=['dataset','seed'],columns='method',values=['mean_set_size','total_shots'])
    dsize=(pairs['mean_set_size']['V2_endpoint']-pairs['mean_set_size']['fixed_certified_1024']).to_numpy()
    dsaved=1-(pairs['total_shots']['V2_endpoint']/pairs['total_shots']['fixed_certified_1024']).to_numpy()
    dsaved_v1=1-(pairs['total_shots']['V2_endpoint']/pairs['total_shots']['V1_adaptive']).to_numpy()
    dsizedom=(pairs['mean_set_size']['V2_rank_upper']-pairs['mean_set_size']['V1_adaptive']).to_numpy()
    assert np.all(dsizedom<=1e-12)

    bg=np.random.default_rng(29017);ind=bg.integers(0,len(dsize),(2000,len(dsize)))
    bootstrap=lambda v:np.quantile(v[ind].mean(axis=1),[.025,.975]).tolist()
    report={
      'date':'2026-11-13','n_runs':args.seeds*3,'n_method_evaluations':len(df),
      'seeds':list(range(args.seed_start,args.seed_start+args.seeds)),
      'alpha':ALPHA,'beta':BETA,'cal_early_frac':CAL_EARLY_FRAC,'test_early_frac':TEST_EARLY_FRAC,
      'set_delta_endpoint_minus_fixed1024_mean':float(dsize.mean()),
      'set_delta_endpoint_minus_fixed1024_bootstrap95':bootstrap(dsize),
      'shots_saved_endpoint_vs_fixed1024_mean':float(dsaved.mean()),
      'shots_saved_endpoint_vs_fixed1024_bootstrap95':bootstrap(dsaved),
      'shots_saved_endpoint_vs_v1_mean':float(dsaved_v1.mean()),
      'shots_saved_endpoint_vs_v1_bootstrap95':bootstrap(dsaved_v1),
      'endpoint_set_no_larger_than_fixed1024_runs':int(np.sum(dsize<=1e-12)),
      'rank_upper_dominates_v1_all_runs':bool(np.all(dsizedom<=1e-12)),
      'endpoint_fixed1024_exact_matches_all_samples':bool(np.all(df[df.method=='V2_endpoint']['endpoint_fixed1024_match_fraction']>=1-1e-12)),
      'audit':'PASS','seconds':float(time.time()-t0),
      'data_notice':'Shot streams pre-generated to pair methods. Counts are hypothetical measured prefixes, not CPU draw counts.',
      'coverage_notice':'Marginal per-query coverage under exchangeability; no hardware-noise or time-varying guarantees.'}
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('\nHELD-OUT RESULTS\n',sm[['method','mean_set','mean_coverage','mean_shots','shots_saved_vs_fixed1024_pct']].to_string(index=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ['seeds','data_notice','coverage_notice']},indent=2))

if __name__=='__main__':main()
