import torch
from pulse.attacker import PairwiseAttacker

def test_feature_dimension():
    model=PairwiseAttacker(16)
    features=model.features(torch.randn(3,16),torch.randn(3,16))
    assert features.shape==(3,68)
    assert model(torch.randn(3,16),torch.randn(3,16)).shape==(3,)
