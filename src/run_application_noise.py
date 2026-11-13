import argparse, io, os, time, json, zipfile, urllib.request, hashlib, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from scipy.stats import t as student_t
import qcp_rank_v2 as q
import experiment_width_matched as w
from v3_reference import calibrated_rho, test_with_spending
from sim_noise import noisy_probabilities

URL='https://archive.ics.uci.edu/static/public/222/bank+marketing.zip'
STAGES=np.asarray(q.STAGES)
NOISE={
 'ideal':(0.,0.,0.),
 'readout_2pct':(0.,0.,.02),
 'gate_depolarizing':(.001,.01,0.),
 'combined_mild':(.001,.01,.02),
 'combined_strong':(.003,.03,.05)
}
METHODS=['Rank+Decision','Width+Decision','Rank+Width','Width+Width','Fixed1024','QCP64']

def bank_dataframe(data_dir):
    root=Path(data_dir);root.mkdir(parents=True,exist_ok=True)
    raw=root/'bank_marketing_official.zip'
    local=os.environ.get('BANK_MARKETING_ZIP')
    if local:
        raw=Path(local)
        if not raw.exists():raise FileNotFoundError(f'BANK_MARKETING_ZIP={local} does not exist')
    if not raw.exists():
        req=urllib.request.Request(URL,headers={'User-Agent':'Mozilla/5.0 (research/1.0)'})
        with urllib.request.urlopen(req,timeout=30) as resp: content=resp.read()
        if not zipfile.is_zipfile(io.BytesIO(content)):raise RuntimeError('UCI download is not a ZIP archive')
        raw.write_bytes(content)
    with zipfile.ZipFile(raw) as outer:
        names=outer.namelist()
        if 'bank.zip' in names:
            with zipfile.ZipFile(io.BytesIO(outer.read('bank.zip'))) as inner:
                data=inner.read('bank-full.csv')
        elif 'bank-full.csv' in names:
            data=outer.read('bank-full.csv')
        else:raise RuntimeError(f'Official archive missing bank-full.csv; files: {names}')
    df=pd.read_csv(io.BytesIO(data),sep=';')
    if len(df)!=45211 or 'y' not in df or 'duration' not in df:
        raise ValueError('Unexpected Bank Marketing format; abort rather than silently substitute data')
    return df,hashlib.sha256(data).hexdigest()

def prepare_bank(df,seed,n_total=1800):

    r=np.random.default_rng(seed+3391)
    if n_total>len(df):n_total=len(df)
    chosen=np.sort(r.choice(len(df),size=n_total,replace=False))
    data=df.iloc[chosen].reset_index(drop=True)
    y=(data['y'].astype(str).str.lower()=='yes').to_numpy(dtype=int)

    X=data.drop(columns=['y','duration'])
    num=list(X.select_dtypes(include=np.number).columns)
    cat=[c for c in X.columns if c not in num]
    idx=np.arange(len(y))
    fit,other=train_test_split(idx,test_size=.5,random_state=seed,stratify=y)
    cal,test=train_test_split(other,test_size=.5,random_state=seed+101,stratify=None)
    try:onehot=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    except TypeError:onehot=OneHotEncoder(handle_unknown='ignore',sparse=False)
    prep=ColumnTransformer([('num',StandardScaler(),num),('cat',onehot,cat)],sparse_threshold=0)
    trainX=prep.fit_transform(X.iloc[fit])

    allX=prep.transform(X)
    proj=PCA(n_components=4,random_state=seed)
    proj.fit(trainX)
    z=proj.transform(allX)
    scale=StandardScaler().fit(z[fit])
    z=np.pi*.78*np.tanh(scale.transform(z)/1.8)
    return (z[fit],y[fit]),(z[cal],y[cal]),(z[test],y[test]),dict(
      n_full=len(df),n_subset=len(data),positive_fraction=float(y.mean()),
      n_positive_cal=int(y[cal].sum()),n_positive_test=int(y[test].sum()),
      excluded_predictor='duration',dim_after_pca=4)

def run_case(name,seed,xy,theta,noise_name,noise,bank_meta=None):
    (zt,yt),(zc,yc),(zx,yx)=xy
    pc=noisy_probabilities(zc,theta,noise)
    px=noisy_probabilities(zx,theta,noise)

    rng=np.random.default_rng(481_900+seed*3211+sum(ord(c) for c in name)+len(noise_name)*103)
    cal=q.sample_prefixes(pc,rng); test=q.sample_prefixes(px,rng)
    old=q.old_calibration(yc,cal)
    qr,_=q.quantile_focused_upper(yc,cal,old['stage'])
    qw,wcost,_=w.width_cal_threshold(yc,cal,.18)
    rho=calibrated_rho(len(yc))
    RD,rds,_=test_with_spending(qr,test,rho)
    WD,wds,_=test_with_spending(qw,test,rho)
    RW,rws,_=w.width_only_test(qr,test,.12,rho)
    WW,wws,_=w.width_only_test(qw,test,.12,rho)
    FIX,fcost,_=q.fixed_certificate(yc,cal,test,1024)
    QCP,qcpcost,_=q.ordinary_qcp(yc,cal,test,64)
    idealq=q.order_threshold(np.where(yc==0,pc,1-pc),old['k'])
    oracle=np.column_stack([px<=idealq,1-px<=idealq])
    combinations={
      'Rank+Decision':(RD,old['shots'],rds),
      'Width+Decision':(WD,wcost,wds),
      'Rank+Width':(RW,old['shots'],rws),
      'Width+Width':(WW,wcost,wws),
      'Fixed1024':(FIX,1024*len(yc),1024*len(yx)),
      'QCP64':(QCP,64*len(yc),64*len(yx))}
    results=[]
    for m,(sets,calcost,testcost) in combinations.items():
        row=dict(dataset=name,seed=seed,noise=noise_name,method=m,
             one_qubit_depol=noise[0],two_qubit_depol=noise[1],readout_flip=noise[2],
             n_train=len(yt),n_cal=len(yc),n_test=len(yx),
             positive_fraction_train=float(yt.mean()),positive_fraction_test=float(yx.mean()),
             shots_cal=int(calcost),shots_test=int(testcost),shots_total=int(calcost+testcost),
             mean_set_size=float(sets.sum(1).mean()),
             observed_coverage=float(sets[np.arange(len(yx)),yx].mean()),
             conditional_positive_coverage=float(sets[yx==1,1].mean()) if np.any(yx==1) else np.nan,
             ideal_containment=float(np.all((~oracle)|sets,axis=1).mean()),
             classifier_accuracy=float(((px>=.5)==yx).mean()),
             oracle_set_size=float(oracle.sum(1).mean()),
             measured_classifier='statevector-trained frozen circuit',
             target='stationary noisy readout mean',
             noise_type='local CPTP gate depolarization and symmetric readout bit-flip',
             is_bank_data_available=bank_meta is not None)
        if bank_meta: row.update(bank_meta)
        results.append(row)
    return results

def aggregate(df):
    g=df.groupby(['dataset','noise','method'],as_index=False,dropna=False).agg(
        shots=('shots_total','mean'),cal_shots=('shots_cal','mean'),test_shots=('shots_test','mean'),
        coverage=('observed_coverage','mean'),size=('mean_set_size','mean'),
        ideal_containment=('ideal_containment','mean'),positive_coverage=('conditional_positive_coverage','mean'),
        class_acc=('classifier_accuracy','mean'),n_runs=('seed','nunique'))
    pv=df.pivot(index=['dataset','seed','noise'],columns='method',values=['shots_total','mean_set_size','observed_coverage'])
    r=(pv['shots_total']['Rank+Decision']/pv['shots_total']['Width+Width'])
    rr=(pv['shots_total']['Rank+Decision']/pv['shots_total']['Width+Decision'])
    saving=pd.DataFrame({'dataset':[x[0] for x in r.index], 'noise':[x[2] for x in r.index],
                         'total_paired_saving_pct':100*(1-r.values),
                         'isolated_cal_saving_pct':100*(1-rr.values)})
    savings=saving.groupby(['dataset','noise'],as_index=False).agg(
        paired_saving_pct=('total_paired_saving_pct','mean'),
        isolated_cal_saving_pct=('isolated_cal_saving_pct','mean'),n_runs=('total_paired_saving_pct','size'))
    return g,savings

def plot(g,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for dataset in g.dataset.unique():
        gg=g[(g.dataset==dataset)&g.method.isin(['Rank+Decision','Width+Decision','Width+Width','QCP64'])]
        fig,ax=plt.subplots(figsize=(8.6,4.7))
        names=list(NOISE)
        for method in ['Rank+Decision','Width+Decision','Width+Width','QCP64']:
            vals=[gg[(gg.noise==z)&(gg.method==method)].shots.iloc[0] if len(gg[(gg.noise==z)&(gg.method==method)]) else np.nan for z in names]
            ax.plot(names,vals,marker='o',label=method)
        ax.set(ylabel='Mean calibration + test shots',xlabel='Stationary noise configuration',title=f'{dataset}: certification measurement cost')
        ax.tick_params(axis='x',rotation=20);ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/f'noise_cost_{dataset}.png',dpi=160);plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--datasets',nargs='+',default=['breast','wine01','digits38','bank'])
    ap.add_argument('--seeds',type=int,nargs='+',default=[60,61,62,63,64])
    ap.add_argument('--epochs',type=int,default=130)
    ap.add_argument('--bank-size',type=int,default=1800)
    ap.add_argument('--bank-required',action='store_true')
    ap.add_argument('--bank-local',default='')
    ap.add_argument('--noise-configs',nargs='+',default=list(NOISE))
    ap.add_argument('--output',default='RESULTS_APPLICATION_NOISE')
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    if args.bank_local: os.environ['BANK_MARKETING_ZIP']=args.bank_local
    assert q.audit_math()=='PASS'
    bank=None;sha=None;failed=[]
    if 'bank' in args.datasets:
        try:
            bank,sha=bank_dataframe(out/'input_data')
            print('UCI Bank Marketing verified:',len(bank),'bank-full.csv SHA256',sha,flush=True)
        except Exception as e:
            if args.bank_required:raise
            failed.append({'dataset':'bank','status':'NOT_RUN','reason':str(e)})
            print('BANK MARKETING NOT EXECUTED:',str(e),flush=True)
    rows=[];t0=time.time()
    for ds in args.datasets:
        if ds=='bank' and bank is None:continue
        for seed in args.seeds:
            if ds=='bank':
                tr,ca,te,meta=prepare_bank(bank,seed,args.bank_size)
            else:
                tr,ca,te=w.prepare_random_holdout(ds,seed);meta=None
            th=q.train(tr,seed,args.epochs)
            for config in args.noise_configs:
                if config not in NOISE:raise ValueError(config)
                noise=NOISE[config]
                rows.extend(run_case(ds,seed,(tr,ca,te),th,config,noise,meta))
            print('Completed',ds,seed,'in',round(time.time()-t0,1),'sec',flush=True)
    if not rows:raise RuntimeError('No dataset executed; refusing empty results')
    df=pd.DataFrame(rows)
    df.to_csv(out/'raw_runs.csv',index=False)
    g,s=aggregate(df);g.to_csv(out/'summary.csv',index=False);s.to_csv(out/'paired_savings.csv',index=False)
    plot(g,out)
    meta=dict(dataset_status={'bank':('RAN' if bank is not None else 'NOT_RUN')},
        bank_sha256=sha,bank_download_url=URL,
        failed_datasets=failed,gate_model='per-RY 1q depolarizing + per-CX 2q depolarizing + symmetric readout flip',
        datasets_executed=sorted(df.dataset.unique().tolist()),seeds=args.seeds,
        noise_configs={key:NOISE[key] for key in args.noise_configs},
        static_calibration_noise_assumption=True,
        target='stationary noisy-readout mean; independent stochastic shots',
        no_hardware_claim=True,train_cost_not_counted=True,wall_seconds=time.time()-t0,
        n_method_rows=len(df),n_dataset_seed_noise_runs=len(df)//len(METHODS))
    (out/'RUN_METADATA.json').write_text(json.dumps(meta,indent=2))
    print('\nSUMMARY\n',g.to_string(index=False));print('\nSAVINGS\n',s.to_string(index=False))
    print('RESULT_DIR',out.resolve())

if __name__=='__main__':main()
