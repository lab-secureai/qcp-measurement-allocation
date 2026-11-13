import numpy as np
from functools import lru_cache
from qcp_rank_v2 import GATE_PAIRS

N=4; DIM=2**N
X=np.array([[0,1],[1,0]],complex)
I=np.eye(2,dtype=complex)

def embed_u(wire,u):
    mats=[u if j==wire else I for j in range(N)]
    ans=mats[0]
    for m in mats[1:]: ans=np.kron(ans,m)
    return ans

@lru_cache(None)
def partial_indices(wires):
    wires=tuple(sorted(wires));keep=tuple(j for j in range(N) if j not in wires)
    idx=np.zeros((2**len(keep),2**len(wires)),int)
    for i in range(idx.shape[0]):
        for j in range(idx.shape[1]):
            b=[0]*N
            for k,w in enumerate(keep):b[w]=(i>>(len(keep)-1-k))&1
            for k,w in enumerate(wires):b[w]=(j>>(len(wires)-1-k))&1
            idx[i,j]=sum(b[w]<<(N-1-w) for w in range(N))
    return idx

def depol(rho,wires,lmbd):
    if lmbd<=0:return rho
    idx=partial_indices(tuple(sorted(wires)))
    idx1=idx.reshape(-1)
    block=rho[np.ix_(idx1,idx1)].reshape(idx.shape[0],idx.shape[1],idx.shape[0],idx.shape[1])
    red=np.einsum('aibi->ab',block)
    out=np.kron(red,np.eye(idx.shape[1],dtype=complex)/idx.shape[1])
    unscramble=np.empty(DIM,dtype=int);unscramble[idx1]=np.arange(DIM)
    mix=out[np.ix_(unscramble,unscramble)]
    return (1-lmbd)*rho+lmbd*mix

@lru_cache(None)
def cnot_perm(c,t):
    return np.array([i^(1<<(N-1-t)) if (i&(1<<(N-1-c))) else i for i in range(DIM)],int)

def ry(theta):
    c=np.cos(theta/2);s=np.sin(theta/2)
    return np.array([[c,-s],[s,c]],dtype=complex)

def prob_density(x,theta,one_gate=0.,two_gate=0.,readout=0.):

    v=np.zeros(DIM,complex);v[0]=1.
    rho=np.outer(v,v.conj())
    def step_ry(r,angle,wire):
        u=embed_u(wire,ry(float(angle)))
        r=u@r@u.conj().T
        return depol(r,(wire,),one_gate)
    def step_cx(r,c,t):
        ix=cnot_perm(c,t)
        r=r[np.ix_(ix,ix)]
        return depol(r,(c,t),two_gate)
    for j in range(4):rho=step_ry(rho,x[j],j)
    for c,t in [(0,1),(1,2),(2,3)]:rho=step_cx(rho,c,t)
    for j in range(4):rho=step_ry(rho,theta[j]+0.45*x[j],j)
    for c,t in [(3,0),(2,0),(1,0)]:rho=step_cx(rho,c,t)
    for j in range(4):rho=step_ry(rho,theta[j+4],j)
    p=float(np.real(np.trace(rho[np.ix_(np.arange(8,16),np.arange(8,16))])))
    return float(np.clip(readout+(1-2*readout)*p,1e-6,1-1e-6))

def noisy_probabilities(z,theta,noise):
    th=theta.detach().cpu().numpy() if hasattr(theta,'detach') else np.asarray(theta)
    return np.array([prob_density(x,th,*noise) for x in z],float)

if __name__=='__main__':
    import torch
    from qcp_rank_v2 import model_p
    rng=np.random.default_rng(10)
    for _ in range(5):
        x=rng.normal(size=4);th=rng.normal(size=8)
        a=prob_density(x,th)
        b=float(model_p(torch.tensor(x[None,:],dtype=torch.float64),torch.tensor(th,dtype=torch.float64))[0])
        assert abs(a-b)<1e-10, (a,b)
        for noise in ((.001,.005,.01),(.002,.02,.03)):
            z=prob_density(x,th,*noise)
            assert 0<=z<=1

    r=np.zeros((16,16),complex);r[0,0]=1
    for wires in [(0,),(1,),(0,1),(1,3)]:
        rr=depol(r,wires,.2)
        assert abs(np.trace(rr)-1)<1e-12
        assert np.linalg.eigvalsh(rr).min()>-1e-12
    print('PASS density matrix simulator, local depolarizing channels, readout errors')
