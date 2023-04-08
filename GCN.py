import torch
import torch_geometric.transforms as T
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, data):
        super(GCN, self).__init__()
        num_features = data.x_t.shape[1]
        self.data = data
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 16)
        
    def encode(self):
        x = self.conv1(self.data.x, self.data.edge_index) 
        x = x.relu()
        return self.conv2(x, self.data.edge_index) 

    def decode(self, z, pos_edge_index, neg_edge_index): # only pos and neg edges
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1) # concatenate pos and neg edges
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  # dot product 
        return logits

    def decode_all(self, z): 
        prob_adj = z @ z.t() # get adj NxN
        return (prob_adj > 0).nonzero(as_tuple=False).t() 
