import math
import numpy as np
from scipy.stats import binom
import qcp_rank_v2 as b
from experiment_width_matched import (width_cal_budgeted, width_cal_threshold,
                                       width_only_test, score_cp_width, ST, NST)

def main():
    rng=np.random.default_rng(985123)
    n=34;cal=rng.binomial(1,np.linspace(.05,.95,n)[:,None],size=(n,1024)).cumsum(1).astype(np.int16)
    labels=rng.integers(0,2,n)
    q,s,st=width_cal_threshold(labels,cal,.22)
    q2,s2,st2=width_cal_threshold(1-labels,cal,.22)
    assert np.array_equal(st,st2) and s==s2, 'width-cal allocation leaked true labels'
    for power in [0.,.5,1.]:
        q3,spent,stage=width_cal_budgeted(labels,cal,10000,power)
        q4,spent2,stage2=width_cal_budgeted(1-labels,cal,10000,power)
        assert spent==spent2 and np.array_equal(stage,stage2), 'width-first allocation leaked true labels'
        assert spent<=10000 and spent>=8*n
    rng=np.random.default_rng(288)
    test=rng.binomial(1,.41,size=(21,1024)).cumsum(1).astype(np.int16)
    a,c,stag_a=width_only_test(.35,test,.12,.25)
    bb,d,stag_b=width_only_test(.85,test,.12,.25)
    assert np.array_equal(stag_a,stag_b) and c==d,'width-only test stopping used qhi'
    assert not np.any(np.diff(ST)<=0)
    assert abs((b.BETA_CAL/(n*NST))*(n*NST)-b.BETA_CAL)<1e-15
    for rho in [.20,.42,.7]:
        weights=np.r_[np.repeat(rho/(NST-1),NST-1),1-rho]
        assert np.all(weights>=0) and abs(weights.sum()-1)<1e-12

    for t in [8,64,256]:
        for p in [0.01,.13,.5,.87,.99]:
            c=np.arange(t+1)
            u=b.one_sided_upper(c,t,.01)
            assert binom.pmf(c,t,p)[u<p-1e-12].sum()<=.0100000001

    for k in [1,7,20,33,34]:
        true=rng.random(n);upper=np.maximum(true,rng.random(n))
        assert b.order_threshold(true,k)<=b.order_threshold(upper,k)+1e-12
    print('PASS: width allocation does not use labels/ranks, test stopping independent of conformal threshold, budgets and CP valid')
if __name__=='__main__':main()
