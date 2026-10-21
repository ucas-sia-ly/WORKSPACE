#!/usr/bin/env python3
"""CCR stage 4 adapter. Scoped instrumentation leaves shared trainer defaults intact."""
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch
import torch
import train_salad
from workflow.ccr4_diagnostics import (auxiliary_rng,local_mode,gradient_snapshot,subtract,accumulate,
    gradient_record,local_patch_record,read_committed,support_digest)
from workflow.local_distillation import LocalDistillationSupervisor
from workflow.synthetic_losses import local_distillation_losses, local_target_weights, asymmetric_multi_positive_infonce
from workflow.diagnostic_gradients import gradient_vector_summary


class Observer:
    def __init__(self,args):
        self.args=args
        self.step=0
        if args.resume:
            from workflow.model import read_checkpoint
            self.step=read_checkpoint(args.resume)['global_step']
        self.path=args.output_dir/'attribution.jsonl'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        rows=read_committed(self.path,self.step)
        self.path.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows))
        self.model=None
        self.named=[]

    def begin(self):
        self.named=[(n,p) for n,p in self.model.named_parameters() if p.requires_grad]
        self.sampled=self.step % self.args.stage4_gradient_interval == 0
        self.grads={}
        self.phase='real'
        self.weather={}
        self.local=[]
        self.aux=None

    def backward(self):
        if not self.sampled:
            return
        if self.phase=='real':
            self.grads['real']=gradient_snapshot(self.named)
        elif self.phase=='syn':
            self.grads['syn_weighted']=subtract(gradient_snapshot(self.named),self.grads['real'])

    def local_grad(self, terms):
        if not self.sampled:
            return
        for name,value in terms.items():
            gradients=torch.autograd.grad(value,[p for _,p in self.named],retain_graph=True,allow_unused=True)
            gradients=[g.detach().float().cpu() if g is not None else None for g in gradients]
            self.grads[name]=accumulate(self.grads.get(name,[None]*len(gradients)),gradients)

    def finish(self):
        record=dict(step=self.step,global_weight=self.args.synthetic_global_weight if self.args.synthetic_aux else 0.,
            weather=self.weather,local_pairs=self.local,gradient_sampled=self.sampled,
            actual_auxiliary_queries=sum(r['queries'] for r in self.weather.values()))
        if self.sampled:
            if self.args.synthetic_aux:
                self.grads.setdefault('syn_weighted',[None]*len(self.named))
            for key,weight in (('feature_weighted',self.args.feature_distill_weight),('ot_weighted',self.args.ot_distill_weight)):
                if weight: self.grads.setdefault(key,[None]*len(self.named))
            record['gradients']=gradient_record(self.grads,self.named)
        with self.path.open('a') as handle:
            handle.write(json.dumps(record,allow_nan=False)+'\n')
        self.grads={}
        self.aux=None
        self.step+=1


def train(args):
    if (args.precision not in ('32','bf16') or args.reliability_ot or args.real_global_distill_weight
            or args.score_distill_weight or args.synthetic_places_only or args.synthetic_fraction != 0
            or args.real_sampling!='coverage_pk' or args.ot_min_teacher_keep != torch.finfo(torch.float32).eps):
        raise ValueError('Stage 4 permits only real MS + declared global/Feature/OT, default numerical OT floor')
    if args.stage4_local_mode not in ('train','salad_dropout_off','eval') or args.stage4_gradient_interval < 1:
        raise ValueError('Invalid diagnostic policy')
    args.local_distill_eval=False
    args.isolate_auxiliary_rng=args.stage4_rng=='isolated'
    observer=Observer(args)
    from workflow import model as model_module, training_data, local_distillation
    original_model=model_module.SALADModel
    original_optimizer=torch.optim.AdamW
    original_scaler=torch.cuda.amp.GradScaler
    original_config=train_salad.resolved_training_config
    original_sample=training_data.SyntheticAuxSampler.sample

    class ObservedModel(original_model):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            observer.model=self

    class ObservedOptimizer(original_optimizer):
        def zero_grad(self,*a,**kw):
            observer.begin()
            return super().zero_grad(*a,**kw)
        def step(self,*a,**kw):
            result=super().step(*a,**kw)
            observer.finish()
            return result

    class Backward:
        def __init__(self,value): self.value=value
        def backward(self,*a,**kw):
            self.value.backward(*a,**kw)
            observer.backward()

    class ObservedScaler(original_scaler):
        def scale(self,value): return Backward(super().scale(value))

    def config(*a,**kw):
        result=original_config(*a,**kw)
        result['stage4']=dict(version=1,rng=args.stage4_rng,rng_algorithm='sha256_seed_step_branch_fork_cpu_active_cuda_v1',
            local_mode=args.stage4_local_mode,gradient_interval=args.stage4_gradient_interval,
            zero_weight_control=args.synthetic_aux and args.synthetic_global_weight==0,
            support='teacher_fixed_A_only_common_support',feature='backbone_local_features',
            dropout_scope='SALAD aggregator Dropout only when salad_dropout_off')
        if args.synthetic_aux:
            result['synthetic_aux']['model_rng']='ccr4_hash_fork_v1' if args.stage4_rng=='isolated' else 'shared_main_torch_rng'
        return result

    def sample(sampler,*a,**kw):
        result=original_sample(sampler,*a,**kw)
        observer.aux=result
        return result

    def global_loss(queries,labels,references,real_labels,temperature=.1,pair_valid=None,denominator=None):
        value=asymmetric_multi_positive_infonce(queries,labels,references,real_labels,temperature,pair_valid,denominator)
        observer.phase='syn'
        with torch.no_grad():
            conditions=observer.aux['condition']
            for condition in sorted(set(conditions)):
                mask=torch.tensor([c==condition for c in conditions],device=queries.device)
                valid=mask if pair_valid is None else mask & pair_valid
                if valid.any():
                    mean=asymmetric_multi_positive_infonce(queries[valid],labels[valid],references,real_labels,temperature)
                    observer.weather[condition]=dict(queries=int(valid.sum()),loss_sum=float(mean)*int(valid.sum()))
        return value

    class Supervisor(LocalDistillationSupervisor):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.provenance.update(student_local_mode=args.stage4_local_mode,model_rng=args.stage4_rng)

        def prepare(self,batch):
            prepared,totals,stats=super().prepare(batch)
            valid=[i for i,v in enumerate(batch['pair_valid']) if bool(v)]
            for pair,i in zip(prepared,valid):
                pair['condition']=batch['condition'][i]
                pair['source_path']=batch['source_paths'][i]
                pair['generated_path']=batch['synthetic_paths'][i]
            return prepared,totals,stats

        def backward(self,model,prepared,totals,scaler,precision,weights,*,global_step=0):
            metrics=dict(feature=0.,score=0.,ot=0.,feature_cosine=0.,weighted_total=0.)
            counter_totals={mode:dict(feature=0.,ot=0.) for mode in ('uniform','verified')}
            if observer.sampled:
                for pair in prepared:
                    verified=pair['positive'].squeeze(1)
                    for mode in counter_totals:
                        conf=(verified>0).float() if mode=='uniform' else verified
                        target_weights=local_target_weights(pair['teacher'],pair['index'],conf)
                        for key in counter_totals[mode]: counter_totals[mode][key]+=float(target_weights[key].sum())
            for pair in prepared:
                teacher={k:v.to(self.device) for k,v in pair['teacher'].items()}
                index,q=pair['index'].to(self.device),pair['q'].to(self.device)
                meta=dict(condition=pair['condition'],source_path=str(pair['source_path']),generated_path=str(pair['generated_path']))
                if not any(weights[k] and pair['masses'][k]>0 for k in weights):
                    observer.local.append(dict(**meta,patches=q.numel(),positive_a_patches=int((q>0).sum()),
                        support_sha256=support_digest(index,q),zero_supervision=True))
                    continue
                with local_mode(model,args.stage4_local_mode), torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=precision=='bf16'):
                    _,student=model(pair['image'].to(self.device),return_aux=True)
                losses=local_distillation_losses(student,teacher,index,q)
                pieces={k:losses[k]*(losses[k+'_mass']+1e-8)/(totals[k]+1e-8) for k in weights}
                terms={k+'_weighted':weights[k]*pieces[k] for k in ('feature','ot') if weights[k]}
                total=sum(terms.values())
                if not torch.isfinite(total): raise RuntimeError('Nonfinite local supervision')
                observer.local_grad(terms)
                if observer.sampled:
                    diagnostic=local_patch_record(student,teacher,index,q)
                    counter_gradients={}
                    verified=pair['positive'].squeeze(1).to(self.device)
                    for mode in counter_totals:
                        confidence=(verified>0).float() if mode=='uniform' else verified
                        counter=local_distillation_losses(student,teacher,index,confidence)
                        total_counter=sum(weights[k]*counter[k]*(counter[k+'_mass']+1e-8)/(counter_totals[mode][k]+1e-8)
                                          for k in ('feature','ot') if weights[k])
                        gradients=torch.autograd.grad(total_counter,[p for _,p in observer.named],retain_graph=True,allow_unused=True)
                        counter_gradients[mode]=[g.detach().float().cpu() if g is not None else None for g in gradients]
                    diagnostic['q_gradient_control']=gradient_vector_summary(counter_gradients,observer.named,real_loss_name='uniform')
                    for band in diagnostic['confidence_bands']:
                        band['weighted_feature_contribution']=weights['feature']*band['feature_numerator']/(totals['feature']+1e-8)
                        band['weighted_ot_contribution']=weights['ot']*band['ot_numerator']/(totals['ot']+1e-8)
                    observer.local.append(dict(**meta,**diagnostic))
                else:
                    observer.local.append(dict(**meta,patches=q.numel(),positive_a_patches=int((q>0).sum()),
                        confidence_mass=float(q.sum()),effective_sample_size=float(q.sum().square()/q.square().sum()) if q.square().sum()>0 else 0.,
                        support_sha256=support_digest(index,q)))
                observer.phase='local'
                scaler.scale(total).backward()
                for k in weights: metrics[k]+=float(pieces[k].detach())
                metrics['feature_cosine']+=float(losses['feature_cosine']*(losses['feature_mass']+1e-8)/(totals['feature']+1e-8))
                metrics['weighted_total']+=float(total.detach())
            return metrics

    with ExitStack() as stack:
        for target,name,value in ((model_module,'SALADModel',ObservedModel),(torch.optim,'AdamW',ObservedOptimizer),
            (torch.cuda.amp,'GradScaler',ObservedScaler),(train_salad,'resolved_training_config',config),
            (training_data.SyntheticAuxSampler,'sample',sample),(local_distillation,'LocalDistillationSupervisor',Supervisor)):
            stack.enter_context(patch.object(target,name,value))
        stack.enter_context(patch('workflow.synthetic_losses.asymmetric_multi_positive_infonce',global_loss))
        stack.enter_context(patch.object(train_salad,'auxiliary_model_rng',lambda a,d,s,b:auxiliary_rng(a.seed,s,b,d,a.stage4_rng=='isolated')))
        return train_salad.train(args)


def parse_args(argv=None):
    import argparse
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--stage4-rng',choices=('original','isolated'),default='isolated')
    p.add_argument('--stage4-local-mode',choices=('train','salad_dropout_off','eval'),default='salad_dropout_off')
    p.add_argument('--stage4-gradient-interval',type=int,default=100)
    extra,rest=p.parse_known_args(argv)
    args=train_salad.parse_args(rest)
    for key,value in vars(extra).items(): setattr(args,key,value)
    return args


if __name__=='__main__': train(parse_args())
