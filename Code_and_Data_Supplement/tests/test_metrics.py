import numpy as np
from pulse.metrics import pair_reid_auc, retrieval_metrics, drop_r

def test_metrics_are_finite():
    rng=np.random.default_rng(7)
    base=rng.normal(size=(8,12)).astype(np.float32)
    embeddings=np.repeat(base,2,axis=0)+.01*rng.normal(size=(16,12))
    patients=np.repeat(np.arange(8),2).astype(str)
    auc=pair_reid_auc(embeddings,patients,n_pairs=400,seed=9)
    retrieval=retrieval_metrics(embeddings,patients,chunk_size=4)
    assert .5 < auc <= 1
    assert 0 <= retrieval["cmc_rank1"] <= 1
    assert 0 <= retrieval["map_at_r"] <= 1
    assert np.isfinite(drop_r(retrieval["map_at_r"],.5))
