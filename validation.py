"""Two independent human coders, consensus ratings and model-distance correspondence."""
from collections import Counter
import json
import numpy as np
import pandas as pd
from common import BLOCKS, check_creators, write_json

def correlation(rows,label='consensus'):
    scores=np.array([r[label] for r in rows],float)
    distances=np.array([r['distances'] for r in rows],float)
    ranks=pd.DataFrame(np.column_stack([scores,distances])).rank(method='average').to_numpy()
    return np.corrcoef(ranks.T)[:3,3:]

def agreement(rows):
    a=np.array([r['coder1'] for r in rows],int)
    b=np.array([r['coder2'] for r in rows],int)
    loss=np.abs(np.arange(5)[:,None]-np.arange(5)[None,:])/4
    kappas=[]; tables=[]
    for dimension in range(3):
        table=np.zeros((5,5),int)
        np.add.at(table,(a[:,dimension]-1,b[:,dimension]-1),1)
        observed=table/len(rows)
        expected=np.outer(observed.sum(1),observed.sum(0))
        denominator=np.sum(loss*expected)
        kappas.append(1-np.sum(loss*observed)/denominator if denominator else np.nan)
        tables.append(table)
    return dict(n=len(rows),linear_weighted_kappa=kappas,exact_agreement=np.mean(a==b,axis=0),
                within_one_point=np.mean(abs(a-b)<=1,axis=0),
                mean_absolute_difference=np.mean(abs(a-b),axis=0),confusion_tables=tables)

def validate(rows):
    if len(rows)!=150 or len({r['pair_id'] for r in rows})!=150: raise ValueError('Expected 150 unique pairs.')
    creators=sorted({r[c] for r in rows for c in ('creator_a','creator_b')})
    check_creators(creators)
    for r in rows:
        for label in ('coder1','coder2'):
            if len(r[label])!=3 or any(v not in (1,2,3,4,5) for v in r[label]):
                raise ValueError('Independent ratings must be complete integers from 1 to 5.')
        complete=all(v is not None for v in r['distances'])
        if complete:
            if any(v not in (1,2,3,4,5) for v in r['consensus']): raise ValueError('Missing consensus rating.')
            if not np.isfinite(r['distances']).all(): raise ValueError('Nonfinite model distances.')
        r['distance_eligible']=complete
    if sum(r['distance_eligible'] for r in rows)!=132: raise ValueError('Expected 132 complete distance pairs.')
    expected={'within_near':47,'within_far':44,'between_same_halfyear':41,
              'diagnostic_embodied_missing':14,'diagnostic_caption_missing':4}
    if Counter(r['stratum'] for r in rows)!=expected: raise ValueError('Unexpected validation strata.')
    return creators

def run(inputs,outputs,bootstrap_draws=10000):
    rows=json.loads((inputs/'human_ratings.json').read_text(encoding='utf-8'))
    creators=validate(rows)
    result={'dimensions':BLOCKS,'n_pairs':len(rows),'agreement':{},'correlations':{},
            'bootstrap_seed':2026101901,'bootstrap_draws':bootstrap_draws}
    for name,subset in [('all150',rows),('complete132',[r for r in rows if r['distance_eligible']]),
                        ('diagnostic18',[r for r in rows if not r['distance_eligible']])]:
        item=agreement(subset)
        loo={c:agreement([r for r in subset if c not in (r['creator_a'],r['creator_b'])]) for c in creators}
        values=np.array([v['linear_weighted_kappa'] for v in loo.values()])
        item.update(leave_one_creator_out=loo,kappa_loco_min=values.min(0),kappa_loco_max=values.max(0))
        result['agreement'][name]=item
    for group in ('within','between'):
        subset=[r for r in rows if r['stratum'].startswith(group+'_')]
        matrix=correlation(subset)
        loo={c:correlation([r for r in subset if c not in (r['creator_a'],r['creator_b'])]) for c in creators}
        item=dict(n=len(subset),rho_matrix=matrix,independent_coder1_matrix=correlation(subset,'coder1'),
            independent_coder2_matrix=correlation(subset,'coder2'),leave_one_creator_out=loo,
            loco_min=np.min(list(loo.values()),axis=0),loco_max=np.max(list(loo.values()),axis=0))
        if group=='within':
            rng=np.random.default_rng(2026101901)
            clusters={c:[r for r in subset if r['creator_a']==c] for c in creators}
            boot=[]; boot_kappa=[]
            for _ in range(bootstrap_draws):
                sample=[r for c in rng.choice(creators,len(creators),replace=True) for r in clusters[c]]
                boot.append(correlation(sample)); boot_kappa.append(agreement(sample)['linear_weighted_kappa'])
            boot=np.array(boot)
            item.update(ci95_lower=np.quantile(boot,.025,axis=0),ci95_upper=np.quantile(boot,.975,axis=0),
                independent_agreement=agreement(subset),independent_kappa_ci95=np.quantile(boot_kappa,[.025,.975],axis=0))
            item['matched_minus_cross']=[dict(rating=BLOCKS[i],cross_distance=BLOCKS[j],
                difference=matrix[i,i]-matrix[i,j],
                ci95=np.quantile(boot[:,i,i]-boot[:,i,j],[.025,.975])) for i in range(3) for j in range(3) if i!=j]
        result['correlations'][group]=item
    write_json(outputs/'human_validation.json',result)

