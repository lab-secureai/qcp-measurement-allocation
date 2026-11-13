from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def close(a, b, tol=1e-8):
    return math.isclose(float(a), float(b), abs_tol=tol, rel_tol=0)


def verify_factorial(folder, expected_seed_start, reference_savings):
    p = RES / 'factorial' / folder
    raw = pd.read_csv(p / 'factorial_all_runs.csv')
    details = pd.read_csv(p / 'factorial_sample_details.csv')
    summary = pd.read_csv(p / 'factorial_summary.csv')
    metrics = json.loads((p / 'factorial_metrics.json').read_text(encoding='utf8'))
    methods = {'R+D', 'W+D', 'R+W', 'W+W', 'Wbudget+D'}
    require(len(raw) == 150, f'{folder}: expected 150 rows')
    require(set(raw.dataset) == {'breast','wine01','digits38'}, f'{folder}: datasets')
    require(set(raw.seed) == set(range(expected_seed_start, expected_seed_start + 10)), f'{folder}: seed range')
    require(set(raw.method) == methods, f'{folder}: methods')
    require(not raw.duplicated(['dataset','seed','method']).any(), f'{folder}: duplicates')
    require(np.allclose(raw.cal_shots + raw.test_shots, raw.total_shots), f'{folder}: shots decomposition')
    require(raw.coverage.between(0,1).all() and raw.mean_set_size.between(0,2).all(), f'{folder}: metrics range')
    require(raw.total_shots.gt(0).all(), f'{folder}: positive shots')
    pvt = raw.pivot(index=['dataset','seed'], columns='method', values=['total_shots','mean_set_size','coverage'])
    def saving(m, b): return 100 * np.mean(1 - pvt['total_shots'][m] / pvt['total_shots'][b])
    s = saving('R+D', 'W+D')
    require(close(s, reference_savings, 1e-10), f'{folder}: isolated paired savings differs: {s}')
    for contrast in metrics['contrasts']:
        require(close(saving(contrast['a'], contrast['b']), contrast['saving_pct'], 1e-10), f'{folder}: metric contrast differs')
    for _, row in summary.iterrows():
        source = raw[raw.method == row['method']]
        require(close(source.total_shots.mean(), row['mean_shots']), f'{folder}: summary mean shots mismatch')
        require(close(source.mean_set_size.mean(), row['mean_set_size']), f'{folder}: summary mean size mismatch')
        require(close(source.coverage.mean(), row['mean_coverage']), f'{folder}: summary coverage mismatch')
    wide = details.pivot(index=['dataset','seed','sample_index'], columns='method', values='set_mask')
    mismatch = int((wide['R+D'] != wide['W+D']).sum())
    if folder == 'unstratified':
        require(len(wide) == 2660, 'Primary 2,660 test sample count not reproduced')
        require(mismatch == 2, 'Primary sample-wise mismatch count not reproduced')
        for comp in metrics['reproduction_checks'].values():
            require(all(abs(float(v)) <= 1e-9 for v in comp.values()), 'Frozen V4 reproduction comparison mismatch')
    return {'seed_range': [expected_seed_start, expected_seed_start+9], 'n_method_rows': len(raw),
            'n_dataset_seed_runs': len(pvt), 'n_test_samples': len(wide), 'rank_vs_width_set_mismatches': mismatch,
            'isolated_rank_saving_pct': float(s), 'pipeline_saving_pct': float(saving('R+D','W+W')),
            'mean_rank_decision_shots': float(raw[raw.method=='R+D'].total_shots.mean()),
            'mean_width_width_shots': float(raw[raw.method=='W+W'].total_shots.mean())}


def verify_noise_suite(name, expected_datasets, expected_seeds):
    folder = RES / name
    raw = pd.read_csv(folder / 'raw_runs.csv')
    summ = pd.read_csv(folder / 'summary.csv')
    savings = pd.read_csv(folder / 'paired_savings.csv')
    meta = json.loads((folder / 'RUN_METADATA.json').read_text(encoding='utf8'))
    expect_n = len(expected_datasets) * len(expected_seeds) * 5 * 6
    require(len(raw) == expect_n, f'{name}: expected {expect_n} rows')
    require(not raw.duplicated(['dataset','seed','noise','method']).any(),f'{name}: duplicates')
    require(set(raw.dataset)==set(expected_datasets) and set(raw.seed)==set(expected_seeds),f'{name}: datasets/seeds')
    require(raw.noise.nunique()==5 and raw.method.nunique()==6,f'{name}: noise/method counts')
    require(np.allclose(raw.shots_cal+raw.shots_test, raw.shots_total), f'{name}: shots decomposition')
    for column in ['observed_coverage','ideal_containment','conditional_positive_coverage','classifier_accuracy']:
        require(raw[column].dropna().between(0,1).all(), f'{name}: {column} range')
    require(raw.mean_set_size.between(0,2).all(),f'{name}: set sizes')
    for _, row in summ.iterrows():
        x=raw[(raw.dataset==row.dataset)&(raw.noise==row.noise)&(raw.method==row.method)]
        require(len(x)==len(expected_seeds),f'{name}: group missing')
        for source,target in [('shots_total','shots'),('observed_coverage','coverage'),('mean_set_size','size'),('ideal_containment','ideal_containment')]:
            require(close(x[source].mean(),row[target],1e-8),f'{name}: bad summary {source}')
    pvt=raw.pivot(index=['dataset','seed','noise'],columns='method',values='shots_total')
    savings_all={}
    for _, row in savings.iterrows():
        g=pvt.xs((row['dataset'],row['noise']),level=('dataset','noise'))
        a=g['Rank+Decision'];w=g['Width+Width'];d=g['Width+Decision']
        s=float(100*(1-a/w).mean()); isolated=float(100*(1-a/d).mean())
        require(close(s,row['paired_saving_pct'],1e-9), f'{name}: paired savings mismatch')
        require(close(isolated,row['isolated_cal_saving_pct'],1e-9), f'{name}: isolated savings mismatch')
        savings_all.setdefault(row.noise,[]).extend((100*(1-a/w)).tolist())
    require(meta['n_method_rows']==len(raw),f'{name}: metadata count')
    if name == 'bank_marketing':
        require(meta['dataset_status']['bank']=='RAN' and meta['bank_sha256'], 'Bank run not confirmed')
        require(raw.excluded_predictor.eq('duration').all() and raw.n_full.eq(45211).all(), 'Bank input checks')
        require(raw.n_subset.eq(1800).all(), 'Bank sample size')
    return {'n_method_rows': len(raw),'n_independent_dataset_seed_noise':len(pvt),
            'dataset_count':len(expected_datasets),'seed_count':len(expected_seeds),
            'paired_saving_pct_by_noise':{k:float(np.mean(v)) for k,v in savings_all.items()},
            'bank_run_provenance_sha256':meta.get('bank_sha256')}


def main():
    report={'factorial_primary':verify_factorial('unstratified',50,14.633387153718283),
            'factorial_stratified':verify_factorial('stratified',40,15.42831934523457),
            'noise':verify_noise_suite('noise', ['breast','wine01','digits38'],list(range(60,70))),
            'bank_marketing':verify_noise_suite('bank_marketing',['bank'],list(range(60,65)))}
    out=ROOT/'reports'/'release_verification.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('PASS: release CSV validation')
    print('Factorial paired rank saving: %.3f%%'%report['factorial_primary']['isolated_rank_saving_pct'])
    print('Factorial pipeline saving: %.3f%%'%report['factorial_primary']['pipeline_saving_pct'])
    print('Noise results:',report['noise']['n_method_rows'],'rows')
    print('Bank Marketing:',report['bank_marketing']['n_method_rows'],'rows')
    print('Report:',out)

if __name__=='__main__':main()
