"""Study 2 assignment comparison, stationary null and disjoint-baseline checks.

The assignment comparison holds observed movements fixed. The stationary null
checks its behavior under finite video sampling; it is not a behavioral model.
"""
import numpy as np
import pandas as pd
from common import stable_seed, write_json
from study2 import transitions, decompose

def assignment(x0,x1,rng,draws=5000):
    a=x0-x0.mean(0)
    e=x1-x0
    e-=e.mean(0)
    h=np.mean(np.sum(e*e,axis=1))
    observed=decompose(x0,x1)['DeltaV']
    values=np.array([2*np.mean(np.sum(a*e[rng.permutation(len(e))],axis=1))+h for _ in range(draws)])
    lower=(1+np.sum(values<=observed))/(draws+1)
    upper=(1+np.sum(values>=observed))/(draws+1)
    return dict(mean=values.mean(),q025=np.quantile(values,.025),q975=np.quantile(values,.975),
                p_lower=lower,p_upper=upper,p_two_sided=min(1,2*min(lower,upper)))

def stationary(arrays0,arrays1,rng,draws=1000,permutations=5000):
    n=len(arrays0)
    features=arrays0[0].shape[1]
    x0=np.empty((draws,n,features)); x1=np.empty_like(x0)
    for i,(a,b) in enumerate(zip(arrays0,arrays1)):
        pool=np.vstack([a,b]); m=len(pool)
        w0=rng.multinomial(len(a),np.full(m,1/m),size=draws)/len(a)
        w1=rng.multinomial(len(b),np.full(m,1/m),size=draws)/len(b)
        x0[:,i]=w0@pool; x1[:,i]=w1@pool
    a=x0-x0.mean(1,keepdims=True)
    e=x1-x0; e-=e.mean(1,keepdims=True)
    c=2*np.einsum('bid,bid->b',a,e)/n
    h=np.einsum('bid,bid->b',e,e)/n
    cross=2*np.einsum('bid,bjd->bij',a,e)/n
    perm=np.array([rng.permutation(n) for _ in range(permutations)])
    reassigned=sum(cross[:,i,perm[:,i]] for i in range(n))
    band=np.quantile(reassigned,.025,axis=1)
    observed=decompose(np.array([a.mean(0) for a in arrays0]),np.array([b.mean(0) for b in arrays1]))
    lower=(1+np.sum(c+h<=observed['DeltaV']))/(draws+1)
    upper=(1+np.sum(c+h>=observed['DeltaV']))/(draws+1)
    return dict(null_C_mean=c.mean(),null_C_q025=np.quantile(c,.025),null_C_q975=np.quantile(c,.975),
        null_C_negative_rate=np.mean(c<0),null_assignment_below95_rate=np.mean(c<band-1e-12),
        null_DeltaV_mean=(c+h).mean(),null_DeltaV_q025=np.quantile(c+h,.025),null_DeltaV_q975=np.quantile(c+h,.975),
        observed_C=observed['C'],observed_DeltaV=observed['DeltaV'],
        null_C_lower_tail=(1+np.sum(c<=observed['C']))/(draws+1),null_DeltaV_two_tail=min(1,2*min(lower,upper)))

def split_baseline(arrays0,arrays1,rng,draws=500):
    x1=np.array([a.mean(0) for a in arrays1])
    values=[]
    def cross(a,baseline):
        starts=a-a.mean(0)
        movements=x1-baseline; movements-=movements.mean(0)
        return 2*np.mean(np.sum(starts*movements,axis=1))
    for _ in range(draws):
        aa=[]; bb=[]
        for x in arrays0:
            indices=rng.permutation(len(x)); k=len(x)//2
            aa.append(x[indices[:k]].mean(0)); bb.append(x[indices[k:]].mean(0))
        a,b=np.array(aa),np.array(bb)
        values.append((cross(a,b)+cross(b,a))/2)
    values=np.array(values)
    return dict(mean=values.mean(),q025=np.quantile(values,.025),q975=np.quantile(values,.975),
                negative_fraction=np.mean(values<0))

def run(cells_by_variant,outputs):
    outputs.mkdir(parents=True,exist_ok=True)
    reassignment=[]
    for variant,cells in cells_by_variant.items():
        for p0,p1,creators in transitions(cells):
            x0=np.array([cells[c,p0]['joint'].mean(0) for c in creators])
            x1=np.array([cells[c,p1]['joint'].mean(0) for c in creators])
            rng=np.random.default_rng(stable_seed('study2_10_reassign',variant,p0,base=314159))
            reassignment.append(dict(variant=variant,start=p0,end=p1,**assignment(x0,x1,rng)))
    pd.DataFrame(reassignment).to_csv(outputs/'assignment_comparison.csv',index=False)
    rng_null=np.random.default_rng(20261018)
    rng_split=np.random.default_rng(20261019)
    null_rows=[]; split_rows=[]
    cells=cells_by_variant['three_block']
    for p0,p1,creators in transitions(cells):
        aa=[cells[c,p0]['joint'] for c in creators]
        bb=[cells[c,p1]['joint'] for c in creators]
        null_rows.append(dict(start=p0,end=p1,n=len(creators),**stationary(aa,bb,rng_null)))
        split_rows.append(dict(start=p0,end=p1,n=len(creators),**split_baseline(aa,bb,rng_split)))
        print(f'Diagnostics completed: {p0} to {p1}',flush=True)
    pd.DataFrame(null_rows).to_csv(outputs/'stationary_null.csv',index=False)
    pd.DataFrame(split_rows).to_csv(outputs/'split_baseline.csv',index=False)
    write_json(outputs/'simulation_settings.json',dict(stationary_seed=20261018,split_seed=20261019,
        stationary_draws=1000,assignment_draws=5000,split_draws=500))

