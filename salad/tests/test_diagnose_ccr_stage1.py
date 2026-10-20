"""End-to-end CPU backward probe without any optimizer or parameter mutation."""
import copy
import unittest
import torch
from torch import nn
import torch.nn.functional as F

from diagnose_ccr_stage1 import branch_gradients,combined_from_gram,uniform_targets


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=nn.Conv2d(3,4,1)
        self.aggregator=nn.Module()
        self.aggregator.score=nn.Conv2d(4,2,1)
        self.aggregator.cluster_features=nn.Conv2d(4,2,1)
        self.aggregator.token_features=nn.Linear(4,2)

    def forward(self,images,return_aux=False):
        f=self.backbone(images);score=self.aggregator.score(f).flatten(2)
        content=self.aggregator.cluster_features(f).mean((2,3))
        descriptor=F.normalize(torch.cat([content,self.aggregator.token_features(f.mean((2,3)))],1),dim=1)
        # Include score dependence as the real aggregator does.
        descriptor=F.normalize(descriptor+score.mean((1,2))[:,None]*.1,dim=1)
        log=torch.cat([score,torch.zeros_like(score[:,:1])],1).log_softmax(1)
        aux=dict(backbone_local_features=f,score_logits=score,log_assignment=log)
        return (descriptor,aux) if return_aux else descriptor


class DiagnosticIntegrationTests(unittest.TestCase):
    def test_all_branch_gradients_preserve_model_and_target_and_zero_grad_slots(self):
        torch.manual_seed(8);model=TinyStudent();before=copy.deepcopy(model.state_dict())
        teacher={key:value.detach().clone().requires_grad_() for key,value in model(torch.randn(1,3,2,2),True)[1].items()}
        index=torch.arange(4)[None];q=torch.tensor([[.2,.8,0.,1.]],requires_grad=True)
        prepared=[dict(image=torch.randn(1,3,2,2),teacher=teacher,index=index,q=q,
                       masses=dict(feature=2.,score=2.,ot=2.))]
        batch=dict(images=torch.randn(32,3,2,2),labels=torch.arange(8).repeat_interleave(4),
            aux=dict(synthetic_images=torch.randn(2,3,2,2),synthetic_labels=torch.tensor([0,1]),
                     pair_valid=torch.ones(2,dtype=torch.bool)))
        named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
        gradients,losses=branch_gradients(model,batch,prepared,dict(feature=2.,score=2.,ot=2.),named,
            torch.device('cpu'),123,'32','train')
        self.assertEqual(set(gradients),{'real','syn','feature','score','ot'})
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses.values()))
        for name,param in model.named_parameters():
            torch.testing.assert_close(param,before[name],rtol=0,atol=0);self.assertIsNone(param.grad)
        self.assertIsNone(q.grad);self.assertTrue(all(v.grad is None for v in teacher.values()))
        for (name,_),gradient in zip(named,gradients['feature']):
            if name.startswith('aggregator'): self.assertIsNone(gradient)
        uniform,totals=uniform_targets(prepared)
        self.assertEqual(totals,dict(feature=3.,score=3.,ot=3.))
        torch.testing.assert_close(uniform[0]['q'],torch.tensor([[1.,1.,0.,1.]]))

    def test_counterfactual_total_has_same_norm_and_clip_as_explicit_sum(self):
        group={'gram':{'real':{'real':9.,'syn':0.},'syn':{'real':0.,'syn':16.}}}
        total=combined_from_gram(group,dict(real=1.,syn=.5))
        self.assertAlmostEqual(total['norm'],13**.5)
        self.assertAlmostEqual(total['vs_real_cosine'],3/13**.5)
        self.assertTrue(total['would_clip'])


if __name__=='__main__': unittest.main()
