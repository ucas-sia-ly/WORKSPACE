"""Opt-in CCR attribution: RNG contexts, common support and diagnostic reductions."""
from contextlib import contextmanager
import hashlib
import json
import torch
from .diagnostic_gradients import gradient_vector_summary


@contextmanager
def auxiliary_rng(seed, step, branch, device, isolated=True):
    if not isolated:
        yield
        return
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    value = int.from_bytes(hashlib.sha256(f'ccr4:{seed}:{step}:{branch}'.encode()).digest()[:8], 'big') % (2**63)
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(value)
        for index in devices:
            torch.cuda.default_generators[index].manual_seed(value)
        yield


@contextmanager
def local_mode(model, mode):
    """SALAD-only Dropout control preserves backbone mode and autograd."""
    if mode not in ('train', 'salad_dropout_off', 'eval'):
        raise ValueError('Unknown local student mode')
    modules = list(model.modules()) if mode == 'eval' else [m for m in model.aggregator.modules() if isinstance(m, torch.nn.modules.dropout._DropoutNd)] if mode == 'salad_dropout_off' else []
    saved = [(m, m.training) for m in modules]
    try:
        for m in modules:
            m.training = False
        yield
    finally:
        for m, training in saved:
            m.training = training


def confidence_pair(positive_a, verified, index):
    """Uniform never promotes B/Unknown; both losses receive identical A support."""
    q = (verified.detach().float() * positive_a.detach().bool()).clone()
    if not torch.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise ValueError('Invalid confidence')
    q = q * (index >= 0)
    return (q > 0).float(), q


def support_digest(index, q):
    return hashlib.sha256(index.detach().cpu().contiguous().numpy().tobytes() + (q > 0).detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def gradient_snapshot(named):
    return [p.grad.detach().float().cpu().clone() if p.grad is not None else None for _,p in named]


def subtract(a, b):
    return [None if x is None and y is None else -y if x is None else x if y is None else x-y for x,y in zip(a,b)]


def accumulate(a, b):
    return [y if x is None else x if y is None else x+y for x,y in zip(a,b)]


def gradient_record(grads, named):
    record = gradient_vector_summary(grads, named)
    for group in record['groups'].values():
        real = group['losses'].get('real', {}).get('raw_norm', 0.)
        syn = group['losses'].get('syn_weighted', {}).get('raw_norm', 0.)
        group['synthetic_real_norm_ratio'] = syn/real if real > 0 else None
        group['ratio_defined'] = real > 0
    return record


def local_patch_record(student, teacher, index, q):
    """Only diagnostics: no extra gating or random draws, no new targets."""
    from .synthetic_losses import _feature_patch_values, _kl_patch_values
    with torch.no_grad(), torch.autocast(device_type=q.device.type, enabled=False):
        feature, cosine, fq = _feature_patch_values(student['backbone_local_features'], teacher['backbone_local_features'], index, q)
        ot, oq, low = _kl_patch_values(student['log_assignment'],teacher['log_assignment'],index,q,True,None)
        positive = q.flatten(1) > 0
        mass, squared = float(q.sum()), float(q.square().sum())
        values = q.flatten()[positive.flatten()]
        similarity = cosine[positive]
        correlation = None
        if len(values)>1 and values.std()>0 and similarity.std()>0:
            correlation=float(torch.corrcoef(torch.stack((values,similarity)))[0,1])
        # Equal-count ranks provide descriptive confidence bands, not thresholds.
        order = torch.argsort(values)
        f, o, w = feature[positive], ot[positive], values
        bands=[]
        for indices in torch.tensor_split(order,4):
            if not len(indices):
                continue
            bands.append(dict(patches=len(indices), q_min=float(w[indices].min()),q_max=float(w[indices].max()),
                feature_numerator=float((f[indices]*fq[positive][indices]).sum()),
                ot_numerator=float((o[indices]*oq[positive][indices]).sum()), mass=float(w[indices].sum())))
        # Same-coordinate whole-grid baseline is labeled separately from matched support.
        sf=student['backbone_local_features'].flatten(2).float()
        tf=teacher['backbone_local_features'].flatten(2).float()
        coordinate=None
        if sf.shape==tf.shape:
            coordinate=float(torch.nn.functional.cosine_similarity(sf,tf,dim=1).mean())
        return dict(patches=q.numel(),positive_a_patches=int(positive.sum()),positive_fraction=float(positive.float().mean()),
            support_sha256=support_digest(index,q), confidence_mass=mass, effective_sample_size=mass*mass/squared if squared else 0.,
            confidence_min=float(values.min()) if len(values) else None,confidence_max=float(values.max()) if len(values) else None,
            feature_cosine_matched_support=float(cosine[positive].mean()) if positive.any() else None,
            confidence_feature_cosine_correlation=correlation,
            feature_cosine_same_coordinate_all=coordinate, confidence_bands=bands,
            ot_low_teacher_mass_patches=int((low & positive).sum()),
            feature_numerator=float((feature*fq).sum()),ot_numerator=float((ot*oq).sum()),
            feature_mass=float(fq.sum()),ot_mass=float(oq.sum()))


def read_committed(path, completed_steps):
    if not path.exists():
        return []
    # A killed process may leave a partial last line or diagnostics after its last checkpoint.
    rows=[]
    for line in path.read_text().splitlines():
        try:
            row=json.loads(line)
        except json.JSONDecodeError:
            break
        if row['step'] < completed_steps:
            rows.append(row)
    return rows


def summarize_attribution(rows):
    """Explicitly distinguish sampled gradients from all-consumed-pair coverage."""
    from .diagnostic_local import _distribution
    gradients={};weather={};q_controls=[]
    for row in rows:
        for group,values in row.get('gradients',{}).get('groups',{}).items():
            for name,loss in values['losses'].items():
                gradients.setdefault(group+'/'+name+'/norm',[]).append(loss['weighted_norm'])
            for names,value in values['cosines'].items():
                if value['defined']: gradients.setdefault(group+'/'+names+'/cosine',[]).append(value['value'])
            if values['ratio_defined']: gradients.setdefault(group+'/syn_real_ratio',[]).append(values['synthetic_real_norm_ratio'])
        for pair in row['local_pairs']:
            target=weather.setdefault(pair['condition'],dict(pairs=0,patches=0,positive_a_patches=0,zero_supervision_pairs=0,
                per_image_coverage=[],confidence_mass=[],effective_sample_size=[],sampled_feature_cosine=[],sampled_q_cosine_correlation=[]))
            target['pairs']+=1;target['patches']+=pair['patches'];target['positive_a_patches']+=pair['positive_a_patches']
            target['zero_supervision_pairs']+=int(pair.get('zero_supervision',False))
            target['per_image_coverage'].append(pair['positive_a_patches']/pair['patches'])
            for key in ('confidence_mass','effective_sample_size'):
                if key in pair: target[key].append(pair[key])
            if pair.get('feature_cosine_matched_support') is not None: target['sampled_feature_cosine'].append(pair['feature_cosine_matched_support'])
            if pair.get('confidence_feature_cosine_correlation') is not None: target['sampled_q_cosine_correlation'].append(pair['confidence_feature_cosine_correlation'])
            if 'q_gradient_control' in pair: q_controls.append(dict(step=row['step'],source_path=pair['source_path'],**pair['q_gradient_control']))
    for target in weather.values():
        target['positive_fraction']=target['positive_a_patches']/max(1,target['patches'])
        for key,value in list(target.items()):
            if isinstance(value,list): target[key]=_distribution(value)
    return dict(sampled_gradient_distributions={k:_distribution(v) for k,v in gradients.items()},local_weather=weather,
                fixed_student_q_gradient_controls=q_controls,
                scope='No loss-ratio gradient inference; undefined zero-real-gradient cosine/ratio omitted, raw rows retain defined flags')
