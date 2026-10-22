import torch
from pulse.token_scoring import rank_normalize

def test_rank_normalize():
    values=torch.tensor([[3.,1.,2.],[0.,2.,1.]])
    expected=torch.tensor([[1.,0.,.5],[0.,1.,.5]])
    assert torch.allclose(rank_normalize(values),expected)
