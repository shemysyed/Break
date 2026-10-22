import numpy as np
import torch
from pulse.attacker import PairwiseAttacker
from pulse.metrics import pair_reid_auc, retrieval_metrics
from pulse.token_scoring import rank_normalize

x=torch.tensor([[3.,1.,2.],[0.,2.,1.]])
assert torch.allclose(rank_normalize(x),torch.tensor([[1.,0.,.5],[0.,1.,.5]]))
attacker=PairwiseAttacker(8); assert attacker.features(torch.randn(4,8),torch.randn(4,8)).shape==(4,36)
rng=np.random.default_rng(1); embeddings=rng.normal(size=(20,8)).astype(np.float32); patients=np.repeat(np.arange(10),2).astype(str)
auc=pair_reid_auc(embeddings,patients,n_pairs=200,seed=2); retrieval=retrieval_metrics(embeddings,patients,chunk_size=8)
print("Smoke test passed",{"pair_auc":auc,**retrieval})
