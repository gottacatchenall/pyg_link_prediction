import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score


# Load utils from other local files
from bipartite_data import BipartiteData
from load_metaweb import load_edgelist
from load_features import load_features 

from GCN import GCN
from GAT import GAT

edge_index = load_edgelist()
x_s, x_t = load_features(edge_index)

data = BipartiteData(edge_index, x_s, x_t)
data.num_nodes = x_s.shape[0] + x_t.shape[0]

data_list = [data, data]
loader = DataLoader(data_list, batch_size=2)
batch = next(iter(loader))

def train(model, optimizer):
    model.train()

    neg_edge_index = negative_sampling(
        edge_index=model.data.train_pos_edge_index, #positive edges
        num_nodes= model.data.num_nodes,
        num_neg_samples=model.data.train_pos_edge_index.size(1)) # number of neg_sample equal to number of pos_edges

    optimizer.zero_grad()
   
    z = model.encode() 
    link_logits = model.decode(z, model.data.train_pos_edge_index, neg_edge_index) # decode
    
    E = model.data.train_pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float)
    link_labels[:model.data.train_pos_edge_index.size(1)] = 1.
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def test_link(model):
    model.eval()
    perfs = []
    pos_edge_index = model.data['test_pos_edge_index']
    neg_edge_index = model.data['test_neg_edge_index']

    z = model.encode() # encode train
    link_logits = model.decode(z, pos_edge_index, neg_edge_index) # decode test or val
    link_probs = link_logits.sigmoid() # apply sigmoid
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return(roc_auc_score(link_labels.cpu(), link_probs.cpu()), average_precision_score(link_labels.cpu(), link_probs.cpu())) 




learningrate = 1e-2
gcn_optim = getattr(torch.optim, "Adam")(gcn.parameters(), lr=learningrate)

t = T.RandomLinkSplit(is_undirected=True)

train_data, val_data, test_data = t(data)
gcn = GCN(data)

for epoch in range(100):
    train_gat_loss = train(gcn,gcn_optim)
    test_roc, test_pr = test_link(gcn)
    print("Test ROC: ", test_roc)
    print("Test PR: ", test_pr)



# there's an approach here where we use GNNs as a form of structural embedding
# where features on nodes are noise and the goal is to learn some representation
# that accurately classifies species ids



