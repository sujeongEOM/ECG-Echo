import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch
import time
'''
Reference
https://honeyjamtech.tistory.com/68
https://www.youtube.com/watch?v=PXOzkkB5eH0
'''
class CustomDataset(Dataset): 
  def __init__(self, scores, weights, np_path):
    traces = np.load(np_path, "r+")
    print(traces.shape)
    print(scores.shape)
    print(weights.shape)
    
    self.traces = torch.from_numpy(traces)
    self.scores = torch.from_numpy(scores)
    self.weights = torch.from_numpy(weights)
    self.n_samples = traces.shape[0]
    
  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return self.n_samples

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    trace = self.traces[idx]
    scores = self.scores[idx]
    weight = self.weights[idx]    
    
    assert trace.shape == (5000, 8)

    return trace, score, weight