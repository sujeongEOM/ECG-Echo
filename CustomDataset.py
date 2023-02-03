import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import os
'''
Reference
https://honeyjamtech.tistory.com/68
https://www.youtube.com/watch?v=PXOzkkB5eH0
'''
class CustomDataset(Dataset): 
  def __init__(self, fname, scores, weights, trace_dir):
    print(fname.shape)
    print(scores.shape)
    print(weights.shape)
    
    self.fname = fname
    self.scores = scores
    self.weights = weights
    self.trace_dir = trace_dir
    self.n_samples = fname.shape[0]
    
  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return self.n_samples

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    trace_path = os.path.join(self.trace_dir, self.fname[idx])
    trace = pd.read_csv(trace_path)
    trace.pop('III')
    trace.pop('aVR')
    trace.pop('aVL')
    trace.pop('aVF')
    trace = trace.to_numpy()
    assert trace.shape == (5000, 8)
    trace = torch.tensor(trace / 1000)

    score = torch.tensor(self.scores[idx])
    weight = torch.tensor(self.weights[idx])    
    
    return trace, score, weight