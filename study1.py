"""Individual change configurations, gateways and creator fixed-effect associations.

Inputs are frozen, analysis-ready measurements for the ten focal creators.
No sample search, account ranking or outcome-dependent inclusion is performed.
"""
import numpy as np
import pandas as pd
from math import factorial
from scipy.stats import fisher_exact
from scipy.stats import t as student_t
from common import BLOCKS, CREATORS, check_creators, require_columns, period_index, read_frame, write_json

def fit_cluster_fe(data, outcome, predictors, target):
    """OLS with creator fixed effects and CR1 creator-clustered uncertainty."""
    d = data[[outcome,'creator_handle',*predictors]].dropna().copy()
    dummies = pd.get_dummies(d.creator_handle.astype(str),drop_first=True,dtype=float)
    x = np.column_stack([np.ones(len(d)),d[predictors].to_numpy(float),dummies.to_numpy()])
    y = d[outcome].to_numpy(float)
    n,k = x.shape
    groups = d.creator_handle.to_numpy()
    g = len(set(groups))
    if g < 2 or n <= k+2 or np.linalg.matrix_rank(x) < k:
        raise ValueError('Insufficient independent information for fixed-effect model.')
    bread = np.linalg.pinv(x.T@x)
    beta = bread@x.T@y
    residual = y-x@beta
    meat = np.zeros((k,k))
    for creator in sorted(set(groups)):
        mask = groups==creator
        score = x[mask].T@residual[mask]
        meat += np.outer(score,score)
    covariance = (g/(g-1))*((n-1)/(n-k))*bread@meat@bread
    j = 1+predictors.index(target)
    se = np.sqrt(max(0,covariance[j,j]))
    stat = beta[j]/se if se else np.nan
    critical = student_t.ppf(.975,g-1)
    return dict(estimate=beta[j],se_cluster=se,t_cluster=stat,
                p_nominal=2*student_t.sf(abs(stat),g-1),
                ci95_lower=beta[j]-critical*se,ci95_upper=beta[j]+critical*se,
                n_events=n,n_creators=g)

def associations(events):
    require_columns(events,['space','window_family','creator_handle','P','M','S',
        'D_start_priorbank','D_event_priorbank','D_follow_priorbank','gap_days','peer_bank_valid'])
    events = events.loc[events.space.eq('three_block')].copy()
    check_creators(events.creator_handle)
    if len(events) != 106: raise ValueError('Expected 106 frozen three-block events.')
    rows = []
    for family in ('pooled','career','temporal'):
        d = events if family=='pooled' else events.loc[events.window_family.eq(family)]
        peer = d.loc[d.peer_bank_valid.astype(str).str.lower().eq('true')]
        specifications = [
            ('configuration_to_continuation', d, 'P', ['M','S','D_start_priorbank','gap_days'], 'S'),
            ('continuation_to_peer_difference', peer, 'D_follow_priorbank',
             ['D_event_priorbank','P','M','S','gap_days'], 'P')]
        for name,frame,outcome,predictors,target in specifications:
            rows.append(dict(window_family=family,relationship=name,
                             **fit_cluster_fe(frame,outcome,predictors,target)))
    return pd.DataFrame(rows)

def configurations(pairs, triples):
    pairs = pairs.loc[pairs.space_variant.eq('three_block')].copy()
    triples = triples.loc[triples.space_variant.eq('three_block')].copy()
    check_creators(pairs.creator_handle)
    check_creators(triples.creator_handle)
    if len(pairs)!=95 or len(triples)!=83: raise ValueError('Expected 95 pairs and 83 three-period sequences.')
    keys = ['creator_handle','period_t','period_t1']
    if pairs.duplicated(keys).any() or triples.duplicated(keys).any():
        raise ValueError('Duplicate creator-period comparisons.')
    pairs['period_index'] = pairs.period_t.map(period_index)
    pairs = pairs.sort_values(['creator_handle','period_index']).reset_index(drop=True)
    squared = pairs[[b+'_distance' for b in BLOCKS]].to_numpy(float)**2
    if not np.isfinite(squared).all() or (squared.sum(1)<=0).any():
        raise ValueError('Undefined movement shares.')
    shares = squared/squared.sum(1,keepdims=True)
    for j,b in enumerate(BLOCKS): pairs[b+'_movement_share'] = shares[:,j]
    pairs['dominant_dimension'] = np.array(BLOCKS)[np.argsort(shares,axis=1)[:,-1]]
    pairs['anchored_change'] = (shares.max(1)>=.5)&(shares.min(1)<=.2)
    signal = pairs[[b+'_signal_ratio' for b in BLOCKS]].to_numpy(float)
    relative = pairs[[b+'_relative_shift' for b in BLOCKS]].to_numpy(float)
    robust = (pairs.dominant_dimension.to_numpy()==np.array(BLOCKS)[signal.argmax(1)]) & (
              pairs.dominant_dimension.to_numpy()==np.array(BLOCKS)[relative.argmax(1)])
    pairs['robust_gateway'] = np.where(robust,pairs.dominant_dimension,'ambiguous')
    pairs['change_identified'] = pairs.pair_evidence_state.eq('position_change_identified')
    prior = pairs.groupby('creator_handle').pair_evidence_state.shift()
    prior_end = pairs.groupby('creator_handle').period_t1.shift()
    pairs['stable_to_change_entry'] = pairs.change_identified & prior_end.eq(pairs.period_t) & ~prior.eq('position_change_identified')
    first = pairs.loc[pairs.change_identified].groupby('creator_handle',sort=False).head(1)
    first = pd.DataFrame({'creator_handle':CREATORS}).merge(first[keys+['robust_gateway']],on='creator_handle',how='left')
    first['entry_detected'] = first.period_t.notna()
    rates = []
    for creator,g in pairs.groupby('creator_handle'):
        sequence = g.dominant_dimension.to_numpy()
        rates.append(dict(creator_handle=creator,anchored_share=g.anchored_change.mean(),
                          switch_rate=np.mean(sequence[1:]!=sequence[:-1]) if len(sequence)>1 else np.nan))
    rates = pd.DataFrame(rates)
    follow = pairs.loc[pairs.change_identified].merge(triples[keys+['retention_beta','stable_pool_retention_mean','triple_evidence_state']],on=keys,how='left')
    follow['retention_excess'] = follow.retention_beta-follow.stable_pool_retention_mean
    summary = dict(pairs=len(pairs),triples=len(triples),anchored_count=int(pairs.anchored_change.sum()),
        pooled_anchored_share=pairs.anchored_change.mean(),creator_balanced_anchored_share=rates.anchored_share.mean(),
        creator_balanced_switch_rate=rates.switch_rate.mean(),identified_changes=int(pairs.change_identified.sum()),
        strict_persistence_count=int(triples.triple_evidence_state.eq('persistent_position_change_supported').sum()))
    counts=first.loc[first.robust_gateway.isin(BLOCKS)].robust_gateway.value_counts()
    n=int(counts.sum()); observed=int(counts.max()) if n else 0
    probability=0.
    for a in range(n+1):
        for b in range(n-a+1):
            c=n-a-b
            if max(a,b,c)>=observed:
                probability+=factorial(n)/(factorial(a)*factorial(b)*factorial(c))*3.**(-n)
    summary['first_entry_uniform_max_category_p_descriptive']=probability if n else None
    return pairs, first, follow, rates, summary

def run(inputs, outputs):
    outputs.mkdir(parents=True,exist_ok=True)
    associations(read_frame(inputs/'events.csv')).to_csv(outputs/'associations.csv',index=False)
    pairs,first,follow,rates,summary = configurations(read_frame(inputs/'pairs.csv'),read_frame(inputs/'triples.csv'))
    next_pairs=pairs[['creator_handle','period_t','change_identified','dominant_dimension']].rename(
        columns={'period_t':'period_t1','change_identified':'next_identified','dominant_dimension':'next_dimension'})
    sequences=pairs.loc[pairs.change_identified].merge(next_pairs,on=['creator_handle','period_t1'],how='left')
    sequences['dimension_switch']=sequences.dominant_dimension.ne(sequences.next_dimension)
    comparison=sequences.loc[sequences.next_identified.eq(True)&sequences.robust_gateway.isin(['audiovisual','conceptual'])]
    table=[]
    for gateway in ('audiovisual','conceptual'):
        g=comparison.loc[comparison.robust_gateway.eq(gateway)]
        table.append([int(g.dimension_switch.sum()),int((~g.dimension_switch).sum())])
    summary['immediate_gateway_switch_table']=table
    summary['immediate_gateway_switch_fisher_p_descriptive']=fisher_exact(table).pvalue if len(comparison) else None
    sequences.to_csv(outputs/'gateway_next_dimension.csv',index=False)
    for name,frame in [('configurations',pairs),('first_observed_gateways',first),
                       ('gateway_followup',follow),('creator_summaries',rates),
                       ('stable_to_change_gateways',pairs.loc[pairs.stable_to_change_entry])]:
        frame.to_csv(outputs/(name+'.csv'),index=False)
    write_json(outputs/'summary.json',summary)
