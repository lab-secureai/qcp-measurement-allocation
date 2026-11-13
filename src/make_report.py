import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
MAIN=ROOT/'RESULTS'; ROBUST=ROOT/'RESULTS_RANDOM_HOLDOUT'

def load(p):
    sm=pd.read_csv(p/'heldout_summary.csv').set_index('method')
    sel=json.loads((p/'frozen_selection.json').read_text())
    met=json.loads((p/'paired_metrics.json').read_text())
    return sm,sel,met

def p(v,d=4):return f'{v:.{d}f}'
def fnum(v):return f'{v:,.0f}'
ma,sa,pa=load(MAIN);mb,sb,pb=load(ROBUST)
order=['rank_V3','width_selected','width_budget_matched','width_test_only','V2_endpoint','standard_QCP64']
name={'rank_V3':'Rank-Aware V3', 'width_selected':'Width-Only Adaptive (development-matched)',
 'width_budget_matched':'Width-First (calibration-budget matched)',
 'width_test_only':'Width-Only test stopping (rank calibration)',
 'V2_endpoint':'V2 Endpoint', 'standard_QCP64':'Standard QCP (64 fixed shots)'}
def markdown_table(sm):
    rows=['| Method | Mean total shots | Empirical coverage | Mean set size | Ideal-set containment |','|---|---:|---:|---:|---:|']
    for k in order:
        r=sm.loc[k];rows.append(f'| {name[k]} | {fnum(r.mean_shots)} | {100*r.mean_coverage:.2f}% | {r.mean_set_size:.4f} | {100*r.mean_ideal_containment:.2f}% |')
    return '\n'.join(rows)

def latex_table(sm, caption, label):
    rows=[r'\begin{table}[t]',r'\centering',r'\small',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Method & Shots & Coverage (\%) & Set size & Oracle contain. (\%) \\',r'\midrule']
    for k in order:
        r=sm.loc[k];title=name[k].replace(' (development-matched)','').replace(' (calibration-budget matched)','').replace(' (rank calibration)','')
        if k=='rank_V3':title=r'\textbf{Rank-Aware V3}'
        rows.append(f'{title} & {r.mean_shots:.0f} & {100*r.mean_coverage:.2f} & {r.mean_set_size:.4f} & {100*r.mean_ideal_containment:.2f} '+r'\\')
    rows +=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
    return '\n'.join(rows)

rank=ma.loc['rank_V3'];width=ma.loc['width_selected'];r2=mb.loc['rank_V3'];w2=mb.loc['width_selected']
def pair_diagnostics(folder):
    x=pd.read_csv(folder/'heldout_all_runs.csv')
    z=x.pivot(index=['dataset','seed'],columns='method',values=['total_shots','mean_set_size','coverage'])
    r=z['total_shots']['rank_V3'].to_numpy();w=z['total_shots']['width_selected'].to_numpy()
    sz=(z['mean_set_size']['width_selected']-z['mean_set_size']['rank_V3']).to_numpy()
    ds=np.asarray(z.index.get_level_values('dataset'))
    unique=list(dict.fromkeys(ds));grp=[np.where(ds==d)[0] for d in unique]
    rng=np.random.default_rng(90913); boots=[]
    for _ in range(2500):
        dss=rng.integers(len(grp),size=len(grp))
        idx=np.concatenate([rng.choice(grp[i],len(grp[i]),replace=True) for i in dss])
        boots.append(float(np.mean(1-r[idx]/w[idx])))
    return dict(saving=float(np.mean(1-r/w)),bootstrap=np.quantile(boots,[.025,.975]),
      set_delta=float(np.mean(sz)),enlarged=int(np.sum(sz>1e-12)),identical=int(np.sum(abs(sz)<1e-12)),
      cov_delta=float(np.mean(z['coverage']['width_selected']-z['coverage']['rank_V3'])))
a=pair_diagnostics(MAIN);br=pair_diagnostics(ROBUST)
samples=pd.read_csv(MAIN/'heldout_sample_details.csv')
wide=samples.pivot(index=['dataset','seed','index'],columns='method',values='set_size')
changed=int(np.sum(wide['rank_V3']!=wide['width_selected']))
num_test=len(wide)
text=f'''# QCP V3 — Additional Adaptive Baselines and Matched-Quality Evaluation

**Date:** 2026-11-13. **Implementation:** `experiment_width_matched.py` and `qcp_rank_v2.py`.

## Executive result

An adaptive non-rank baseline chosen exclusively on development seeds 0–4 needs **{width.mean_shots:.0f}** shots per held-out run compared with **{rank.mean_shots:.0f}** for rank-aware V3 on seeds 40–49 (3 datasets × 10 seeds), with **identical mean empirical coverage** ({rank.mean_coverage*100:.2f}%) and essentially matched mean set sizes ({rank.mean_set_size:.4f} versus {width.mean_set_size:.4f}).

For a more rigorous conformal experiment, a second replication randomly allocates calibration and test examples from a held-out pool without stratifying on labels. On fresh seeds 50–59, rank-aware uses **{r2.mean_shots:.0f}** shots versus **{w2.mean_shots:.0f}** for width-only, with common observed coverage **{r2.mean_coverage*100:.2f}%** and set sizes **{r2.mean_set_size:.4f}** versus **{w2.mean_set_size:.4f}**.

**Important:** these results provide empirical evidence that conformal-rank information makes this particular certificate more shot-efficient than the studied width-only rules. They do not establish global optimality or superiority to every adaptive QCP method. The theoretical 90% marginal guarantee relies on exchangeability and the simultaneous confidence-bound error budgets.

## Setup

- Model: trained 4-qubit RY/CNOT classifier, statevector simulator, 130 epochs.
- Datasets: Breast Cancer, Wine 0/1, Digits 3/8 from scikit-learn; 30 held-out fitted models per split protocol.
- Predeclared shot stages: 8,16,32,64,128,256,512,1024; same pre-generated potential Bernoulli shot streams for paired measurement-policy comparison.
- Target coverage 1-alpha=90%, alpha=0.10, confidence failure beta=0.01, split into beta_cal=.005 and beta_test=.005.
- Reported cost is total calibration+test measurement shots. Pre-generation of potential measurement streams is solely a paired Monte Carlo technique, not actual hardware cost.
- Development seeds 0–4 used to choose global thresholds from 7 calibration and 5 test CI-width candidates (35 pairs). **No held-out labels are used for method selection.**
- Thresholds selected in both split protocols: calibration CI width <= **{sa['cal_width_threshold']:.2f}** and test CI width <= **{sa['test_width_threshold']:.2f}**.
- Dev size matching tolerance: at most +{sa['set_tolerance']:.3f} labels relative to rank V3, plus nominal empirical coverage >=.90 and ideal-set containment >=.99. This was checked only on development seeds. Test results report actual violations rather than re-tuning.
- Additional matched calibration-budget counterfactual: width-first greedy measurement scheduling up to V3 calibration cost, with no conformal rank in allocation choices. Priority grid width/(incremental shots)^r for r in {{0,0.5,1}}, tuned exclusively on development seeds; **r={sa['budget_selected_power']}** was selected. This paired budget is supplied from V3 and is not a deployable standalone budget choice.

## Main results: original stratified protocol, held-out seeds 40–49

{markdown_table(ma)}

Rank-aware saves **{100*(1-rank.mean_shots/width.mean_shots):.2f}% by ratio of aggregate mean shots**, or **{100*a['saving']:.2f}% as the average paired per-run fraction** (the recommended estimator), versus the development-selected width-only policy. Hierarchical dataset/seed bootstrap 95% descriptive interval for paired per-run saving is [{100*a['bootstrap'][0]:.2f}%, {100*a['bootstrap'][1]:.2f}%]. Width-only set size was larger in **{a['enlarged']}/30** runs, affecting **{changed}/{num_test:,}** test samples; the mean size difference is +{a['set_delta']:.5f}. Coverage is **exactly the same** per run for the two methods in this experiment.

The width-first calibration-budget-matched counterfactual consumes virtually the same total shots ({ma.loc['width_budget_matched','mean_shots']:.0f} vs {rank.mean_shots:.0f}) while producing a mean set **{ma.loc['width_budget_matched','mean_set_size']:.4f}** vs **{rank.mean_set_size:.4f}**, a paired difference of +0.02852. Its observed coverage is higher because it retains more labels; it should not be described as equal set quality.

## Robustness: unstratified calibration/test assignment, fresh seeds 50–59

{markdown_table(mb)}

Rank-aware saves **{100*br['saving']:.2f}%** of shots on average per dataset/seed run compared with the selected width-only adaptive baseline (hierarchical bootstrap descriptive 95% interval [{100*br['bootstrap'][0]:.2f}%, {100*br['bootstrap'][1]:.2f}%]), with identical mean empirical coverage {100*r2.mean_coverage:.2f}% and +{br['set_delta']:.5f} mean set-size difference for width-only. The two methods have the same mean set size in **{br['identical']}/30** runs.

This robustness protocol does not stratify the calibration-versus-test assignment by class: it better matches the conditional exchangeability assumption after a trained classifier and a held-out pool are fixed. The original results were retained solely for apples-to-apples comparison with V3; relying exclusively on the original stratified split would be less rigorous for a conformal coverage theorem.

## What “matched quality” means here

1. **Same certified target:** the four methods Rank V3, Width-Only, Width-First, Width-Test-Only use the same alpha, beta, confidence-bound family, and maximum shots. Their ideal-score set containment certificate is distinct from standard finite-shot QCP.
2. **Same empirical coverage:** Rank V3 and Width-Only Adaptive have the same observed coverage in both independent held-out protocols (not guaranteed identically on every possible dataset).
3. **Near-matched mean set size:** mean size gap is +{a['set_delta']:.5f} (stratified) and +{br['set_delta']:.5f} (random holdout); per-run equality is not universal. Threshold selection used the development split only.
4. **Same calibration budget control:** width-first reaches almost identical calibration shot count but yields larger sets, isolating the benefit of rank-sensitive allocation. Its run-specific budget comes from Rank V3, so this is an analysis-only counterfactual.
5. **Standard QCP 64:** included as a transparent cheap non-certified-ideal-set comparison. Standard finite-shot conformal can still have ordinary marginal coverage, but it does not assert inclusion of the *ideal-probability conformal set*. It must not be called invalid or plotted as an equivalent oracle-containment guarantee.

## Paper conclusion

This baseline study gives a much clearer algorithmic contribution for a KSE/SOICT-style submission: rank-sensitive sampling makes **certified ideal-score QCP** more measurement-efficient than matched-width alternatives in these simulations. However, the theoretical coverage and confidence tools are established methods; the principal new claim remains the algorithm and the paired empirical cost/quality comparison. The 3 datasets use classical 4-qubit simulations, and the comparison does not include all recent adaptive QCP systems. A conference paper is worth drafting, but novelty and scope require careful SOTA positioning. Do not claim a quantum advantage, a novel optimality theorem, or universal shot savings.

## Reproduction and audits

Run `python audit_width.py` first; then:

```bash
python experiment_width_matched.py --split-policy original_stratified --dev-seeds 5 --eval-seeds 10 --eval-start 40 --epochs 130 --output RESULTS
python experiment_width_matched.py --split-policy random_holdout --dev-seeds 5 --eval-seeds 10 --eval-start 50 --epochs 130 --output RESULTS_RANDOM_HOLDOUT
```

All common parameters and frozen selections are documented in `frozen_selection.json` and every dataset/seed result in `heldout_all_runs.csv`. Confidence intervals are descriptive hierarchical bootstraps over only three data sources and should not be treated as population-level inference over arbitrary QML applications.
'''
(ROOT/'RESULTS_REPORT.md').write_text(text)
tex=latex_table(ma,'Original V3 protocol, held-out seeds 40--49.','tab:width_v3')+'\n\n'+latex_table(mb,'Unstratified calibration--test split, fresh seeds 50--59.','tab:width_v3_random')+'\n'
(ROOT/'PAPER_TABLES.tex').write_text(tex)
(ROOT/'METHOD_AND_THEOREM.md').write_text(r'''# Width-aware baselines and theoretical audit

## What the policies do

**Rank-Aware V3:** The V3 calibration allocator uses the order statistic of upper and lower confidence limits to stop sampling scores deemed irrelevant to the quantile. Test sampling stops when two label membership statuses are resolved relative to the upper certified threshold. Its error spending rho(n) is fixed by the public calibration sample size.

**Width-Only Adaptive:** Calibration sampling of instance i proceeds from shot stage t to the next stage *if and only if* the current Clopper–Pearson confidence interval for measurement probability has width above a fixed, development-selected threshold. Test sampling uses the analogous width-only stopping criterion; it does not inspect the conformal threshold for decisions to continue. The conformal quantile is still computed, necessarily, after calibration to construct a conformal prediction set. Thus “rank-free” means **measurement allocation**, not absence of a conformal quantile.

**Width-First Budget-Matched:** Every calibration sample receives 8 shots; each additional block goes to the sample with the largest current interval width divided by (incremental cost)^r, where r belongs to the dev-only candidate grid. Its total calibration budget is copied from the rank-aware method on the same paired stream. This evaluates allocation *conditional on a common oracle budget*, not a standalone deployment choice.

**Width-Only Test:** Rank-Aware calibration is kept fixed; test measurements are adaptively stopped using only interval width. This ablation isolates the contribution of threshold-aware decisions at inference.

## Coverage claim (not an original theorem)

Fix a trained classifier independent of calibration and test pairs. Let the ideal true-label score s_i be a bounded Bernoulli-mean function, and let k=ceil((n+1)(1-alpha+beta)). Set q_* equal to the kth order statistic of ideal calibration scores (or 1 if k=n+1). Let U_i be an upper confidence bound on the true score from finitely many, possibly adaptively stopped, measurements. For every calibration i and every predetermined shot stage j, define a one-sided binomial upper bound with nominal error eta_ij>=0, and require sum_ij eta_ij <= beta_cal. For test query x, define confidence bounds L_y on each candidate-label score with simultaneous stage failure at most beta_test, and require beta_cal+beta_test <= beta.

By union bound, with probability >=1-beta_cal all s_i <= U_i regardless of the stopping policy (provided stagewise bounds and allowances are predeclared, independent of the current unseen observations). Monotonicity of kth order statistics gives q_* <= q_U=U_(k). On the test event of probability >=1-beta_test, L_y <= s(x,y) for all candidate labels. Therefore C(x)={y:L_y<=q_U} contains the ideal conformal set C_*(x)={y:s(x,y)<=q_*} on the intersection. Hence P{C_* subset C_hat} >= 1-beta. Split conformal exchangeability gives P{Y in C_*} >= 1-alpha+beta, so P{Y in C_hat} >= 1-alpha.

**Important distinction:** The above is marginal over both the calibration/test data and the measurement randomness. It is neither distribution-free conditional coverage nor a simultaneous guarantee over unlimited future queries without additional accounting. Standard QCP scores computed at 64 shots have their own marginal coverage argument under exchangeability but not the same ideal-set containment certificate.

## Design validity and fair comparison

- Calibration/test measurements share exactly the same potential Bernoulli shot streams per data point. Cost counts prefixes consumed by each policy rather than full simulated streams.
- No baseline confidence threshold is optimized on held-out seeds 40--49 or 50--59. A global parameter pair is selected from 35 width-only candidates on 15 dev runs (seeds 0--4; three datasets).
- The nominal confidence error budgets and maximum shot counts are identical. The stagewise union bound permits optional stopping and varying per-input counts.
- The original 2026 V3 splits were stratified by labels into calibration and test. The additional robustness evaluation uses a random 50:50 calibration--test split without label stratification in the held-out pool, and is preferable for invoking exchangeability.
- The benefit is empirical for this particular quantum circuit and data; it does not prove uniform superiority of rank-aware versus width-only allocation, nor a quantum computational advantage.
'''.replace('\u007f','\\widehat '))
print('Wrote reports and LaTeX')
