"""Common-period positions, dispersion decomposition and representation sensitivity."""
import numpy as np
import pandas as pd
from common import check_creators, next_period, period_index, read_frame, write_json, stable_seed

COVERAGE = {'minimal':(3,3,14), 'moderate':(5,3,30), 'strong':(8,5,60)}

def dispersion(x):
    return float(np.mean(np.sum((x-x.mean(0))**2,axis=1)))

def decompose(x0,x1):
    starts = x0-x0.mean(0)
    movements = x1-x0
    centered = movements-movements.mean(0)
    c = 2*np.mean(np.sum(starts*centered,axis=1))
    h = np.mean(np.sum(centered**2,axis=1))
    v0,v1 = dispersion(x0),dispersion(x1)
    lengths = np.linalg.norm(movements,axis=1)
    units = movements[lengths>1e-12]/lengths[lengths>1e-12,None]
    cosine = units@units.T
    drift = float(np.linalg.norm(movements.mean(0)))
    if not np.isclose(c+h,v1-v0,atol=1e-12): raise ArithmeticError('Decomposition identity failed.')
    return dict(V_t=v0,V_t1=v1,C=c,H=h,DeltaV=v1-v0,
                common_drift=drift,mean_individual_move=float(lengths.mean()),
                common_drift_ratio=drift/lengths.mean() if lengths.mean()>0 else np.nan,
                mean_pairwise_direction_cosine=float(cosine[np.triu_indices(len(units),1)].mean()) if len(units)>1 else np.nan)

def correction(arrays0,arrays1):
    """Finite-video correction, assuming independent video sampling within cells."""
    centers = [np.array([a.mean(0) for a in arrays]) for arrays in (arrays0,arrays1)]
    terms = []
    for arrays in (arrays0,arrays1):
        variances = [np.sum((a-a.mean(0))**2)/(len(a)*(len(a)-1)) for a in arrays]
        terms.append((1-1/len(arrays))*np.mean(variances))
    result = decompose(*centers)
    t0,t1 = terms
    result.update(T0=t0,T1=t1,C_corrected=result['C']+2*t0,
                  H_corrected=result['H']-t0-t1,DeltaV_corrected=result['DeltaV']+t0-t1)
    if not np.isclose(result['C_corrected']+result['H_corrected'],result['DeltaV_corrected'],atol=1e-12):
        raise ArithmeticError('Corrected decomposition identity failed.')
    return result

def load_inputs(inputs):
    meta = read_frame(inputs/'videos.csv')
    check_creators(meta.creator_handle)
    if len(meta)!=1690 or not meta.content_id.is_unique: raise ValueError('Expected 1,690 unique focal videos.')
    meta['created_at'] = pd.to_datetime(meta.created_at,utc=True,format='mixed')
    meta['period'] = meta.created_at.map(lambda t:f'{t.year}H{1 if t.month<=6 else 2}')
    coordinates = {}
    for name in ('three_block','two_block'):
        with np.load(inputs/('coordinates_'+name+'.npz'),allow_pickle=False) as z:
            coordinates[name] = {k:z[k] for k in z.files}
        z = coordinates[name]
        z['content_id'] = z['content_id'].astype(str)
        if len(set(z['content_id']))!=len(z['content_id']): raise ValueError('Duplicate coordinate identifiers.')
        # A larger private coordinate store may supply reference vectors. Only focal IDs are used.
        keep = np.isin(z['content_id'],meta.content_id)
        coordinates[name] = {k:a[keep] for k,a in z.items() if k=='content_id' or (a.ndim>=1 and len(a)==len(keep))}
        if not np.isfinite(coordinates[name]['joint']).all(): raise ValueError('Nonfinite coordinates.')
    if len(coordinates['three_block']['content_id'])!=1632 or len(coordinates['two_block']['content_id'])!=1681:
        raise ValueError('Coordinate coverage differs from the frozen focal inventory.')
    if not set(coordinates['three_block']['content_id'])<=set(coordinates['two_block']['content_id']):
        raise ValueError('Three-block complete videos must also have two-block representations.')
    for name,z in coordinates.items():
        blocks = ('conceptual','audiovisual','embodied') if name=='three_block' else ('conceptual','audiovisual')
        expected = np.concatenate([z['block_'+b] for b in blocks],axis=1)/np.sqrt(len(blocks))
        if not np.allclose(z['joint'],expected,atol=1e-6): raise ValueError('Inconsistent equal block weighting.')
    return meta, coordinates

def build_cells(meta,coordinates,rule='moderate'):
    complete = set(coordinates['three_block']['content_id'])
    result = {}
    for variant in ('three_block','two_block_matched','two_block_full'):
        z = coordinates['three_block' if variant=='three_block' else 'two_block']
        lookup = {v:i for i,v in enumerate(z['content_id'])}
        allowed = complete if variant!='two_block_full' else set(lookup)
        frame = meta.loc[meta.content_id.isin(allowed)&meta.period.between('2020H1','2026H1')]
        cells = {}
        min_videos,min_days,min_span = COVERAGE[rule]
        for key,g in frame.groupby(['creator_handle','period']):
            g = g.sort_values(['created_at','content_id'])
            span = (g.created_at.max()-g.created_at.min()).total_seconds()/86400
            if len(g)<min_videos or g.created_at.dt.date.nunique()<min_days or span<min_span: continue
            indices = [lookup[v] for v in g.content_id]
            cells[key] = {'ids':tuple(g.content_id), 'joint':z['joint'][indices].astype(float)}
            for block in ('conceptual','audiovisual','embodied'):
                if 'block_'+block in z: cells[key][block] = z['block_'+block][indices].astype(float)
        result[variant] = cells
    a,b = result['three_block'],result['two_block_matched']
    if {k:v['ids'] for k,v in a.items()}!={k:v['ids'] for k,v in b.items()}:
        raise ValueError('Matched representation must use identical videos and membership.')
    return result

def transitions(cells):
    for p0 in sorted({p for _,p in cells},key=period_index):
        p1 = next_period(p0)
        creators = sorted(c for c,p in cells if p==p0 and (c,p1) in cells)
        if len(creators)>=3: yield p0,p1,creators

def neighbor_sets(x,k):
    distances = np.sum(x*x,1)[:,None]+np.sum(x*x,1)[None,:]-2*x@x.T
    np.fill_diagonal(distances,np.inf)
    return [set(np.argsort(row)[:k]) for row in distances]

def neighbor_retention(x0,x1,k):
    return np.mean([len(a&b)/k for a,b in zip(neighbor_sets(x0,k),neighbor_sets(x1,k))])

def neighbor_continuation(x0,x1,x2,k):
    sets = [neighbor_sets(x,k) for x in (x0,x1,x2)]
    new_retained=[]; old_retained=[]
    for a,b,c in zip(*sets):
        new=b-a; old=a&b
        if new: new_retained.append(len(new&c)/len(new))
        if old: old_retained.append(len(old&c)/len(old))
    return dict(new_neighbor_retained_at_t2=np.mean(new_retained) if new_retained else np.nan,
                new_neighbor_creators_with_new_neighbors=len(new_retained),
                original_retained_neighbor_survives_t2=np.mean(old_retained) if old_retained else np.nan,
                t1_neighbor_retention_at_t2=neighbor_retention(x1,x2,k))

def analyze(cells_by_variant):
    rows,neighbors,triples,loo,neighbor_triples = [],[],[],[],[]
    for variant,cells in cells_by_variant.items():
        for p0,p1,creators in transitions(cells):
            aa = [cells[c,p0]['joint'] for c in creators]
            bb = [cells[c,p1]['joint'] for c in creators]
            x0,x1 = np.array([a.mean(0) for a in aa]),np.array([b.mean(0) for b in bb])
            result = correction(aa,bb)
            row = dict(variant=variant,start=p0,end=p1,n=len(creators),creators='|'.join(creators),**result)
            for block in ('conceptual','audiovisual','embodied'):
                if block in cells[creators[0],p0]:
                    r = decompose(np.array([cells[c,p0][block].mean(0) for c in creators]),
                                  np.array([cells[c,p1][block].mean(0) for c in creators]))
                    row.update({block+'_'+k:v for k,v in r.items() if k in ('C','H','DeltaV')})
            rows.append(row)
            for k in (1,2,3):
                if k>=len(creators): continue
                retention = neighbor_retention(x0,x1,k)
                chance = k/(len(creators)-1)
                interpretable = len(creators)>=2*k+2
                neighbors.append(dict(variant=variant,start=p0,end=p1,n=len(creators),k=k,
                    retention=retention,chance_retention=chance,interpretable=interpretable,
                    chance_adjusted_excess=(retention-chance)/(1-chance) if interpretable else np.nan))
            if variant=='three_block' and len(creators)>=4:
                for j,c in enumerate(creators):
                    r = correction(aa[:j]+aa[j+1:],bb[:j]+bb[j+1:])
                    loo.append(dict(start=p0,end=p1,omitted_creator=c,**r))
            p2 = next_period(p1)
            same = [c for c in creators if (c,p2) in cells]
            if len(same)>=3:
                xs = [np.array([cells[c,p]['joint'].mean(0) for c in same]) for p in (p0,p1,p2)]
                vs = [dispersion(x) for x in xs]
                if variant=='three_block':
                    triples.append(dict(start=p0,middle=p1,end=p2,n=len(same),creators='|'.join(same),
                        V_t=vs[0],V_t1=vs[1],V_t2=vs[2],DeltaV_01=vs[1]-vs[0],DeltaV_02=vs[2]-vs[0]))
                for k in (1,2,3):
                    if k>=len(same): continue
                    neighbor_triples.append(dict(variant=variant,start=p0,middle=p1,end=p2,n=len(same),
                        k=k,interpretable=len(same)>=2*k+2,**neighbor_continuation(*xs,k)))
    return [pd.DataFrame(r) for r in (rows,neighbors,triples,loo,neighbor_triples)]

def run(inputs,outputs):
    outputs.mkdir(parents=True,exist_ok=True)
    meta,coordinates = load_inputs(inputs)
    cells = build_cells(meta,coordinates)
    for name,frame in zip(('transitions','neighbors','three_period_comparisons','leave_one_creator_out','neighbor_continuation'),analyze(cells)):
        frame.to_csv(outputs/(name+'.csv'),index=False)
    sensitivity = []
    for rule in COVERAGE:
        variants = build_cells(meta,coordinates,rule)
        frame = analyze(variants)[0]
        for variant,g in frame.groupby('variant'):
            sensitivity.append(dict(coverage_rule=rule,variant=variant,cells=len(variants[variant]),
                transitions=len(g),mean_DeltaV=g.DeltaV.mean(),mean_C=g.C.mean(),mean_H=g.H.mean()))
    pd.DataFrame(sensitivity).to_csv(outputs/'coverage_sensitivity.csv',index=False)
    write_json(outputs/'coverage.json',{name:len(c) for name,c in cells.items()})
    return cells
