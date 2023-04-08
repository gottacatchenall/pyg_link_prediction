import pandas as pd
import networkx as nx
import torch
import numpy as np

def load_features(edge_index):
    df = pd.read_csv("data/features.csv")
    
    x_bee = []
    x_plant = []
    
    for i, x in enumerate(edge_index):
        ids = (x.unique())
        for id in ids:
            r = df[df["node_id"] == id.item()]
            v = np.float32(r.to_numpy()[0][2:])            
            
            if i == 0:
                x_bee.append(v)
            else:
                x_plant.append(v)
        
    return torch.tensor(np.array(x_bee)), torch.tensor(np.array(x_plant))
    
