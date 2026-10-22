from __future__ import annotations
import argparse
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

parser=argparse.ArgumentParser(); parser.add_argument("--embeddings",required=True,help="NPZ with embeddings and patient_ids"); parser.add_argument("--train-pairs",type=int,default=50000); parser.add_argument("--test-pairs",type=int,default=50000); parser.add_argument("--seed",type=int,default=1212); args=parser.parse_args()
payload=np.load(args.embeddings); embeddings=payload["embeddings"].astype(np.float32); embeddings/=np.clip(np.linalg.norm(embeddings,axis=1,keepdims=True),1e-8,None); patients=payload["patient_ids"].astype(str); groups=defaultdict(list)
for i,p in enumerate(patients): groups[p].append(i)
repeated=[p for p,v in groups.items() if len(v)>=2]; all_patients=list(groups); rng=np.random.default_rng(args.seed)
def features(a,b):
 d=a-b; return np.concatenate([a,b,np.abs(d),a*b,[a@b,np.abs(d).mean(),np.linalg.norm(d),(d*d).mean()]])
def pairs(n):
 x=[]; y=[]
 for _ in range(n//2):
  p=repeated[int(rng.integers(len(repeated)))]; i,j=rng.choice(groups[p],2,replace=False); x.append(features(embeddings[i],embeddings[j])); y.append(1)
 for _ in range(n-n//2):
  p,q=rng.choice(all_patients,2,replace=False); i=rng.choice(groups[p]); j=rng.choice(groups[q]); x.append(features(embeddings[i],embeddings[j])); y.append(0)
 return np.asarray(x),np.asarray(y)
x_train,y_train=pairs(args.train_pairs); x_test,y_test=pairs(args.test_pairs); model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced",random_state=args.seed)); model.fit(x_train,y_train); score=model.predict_proba(x_test)[:,1]; print({"learned_pair_attacker_auc":float(roc_auc_score(y_test,score)),"train_pairs":len(y_train),"test_pairs":len(y_test)})
