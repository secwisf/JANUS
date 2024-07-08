import torch
import dgl
from torch.optim import SparseAdam
from torch.utils.data import DataLoader
from dgl.nn.pytorch import MetaPath2Vec

# Define a model
g = dgl.heterograph({
    ('user', 'uc', 'company'): dgl.rand_graph(3, 2).edges(),
    # ('user', 'uc1', 'company'): dgl.rand_graph(20, 10).edges(),
    ('company', 'cp', 'product'): dgl.rand_graph(2, 0).edges(),
    ('company', 'cu', 'user'): dgl.rand_graph(2, 1).edges(),
    ('product', 'pc', 'company'): dgl.rand_graph(2, 1).edges()
})
model = MetaPath2Vec(g, ['uc', 'cu'], window_size=1)

# Use the source node type of etype 'uc'
dataloader = DataLoader(torch.arange(g.num_nodes('user')), batch_size=32,
                        shuffle=True, collate_fn=model.sample)
optimizer = SparseAdam(model.parameters(), lr=0.025)

for (pos_u, pos_v, neg_v) in dataloader:
    loss = model(pos_u, pos_v, neg_v)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Get the embeddings of all user nodes
user_nids = torch.LongTensor(model.local_to_global_nid['user'])
user_emb = model.node_embed(user_nids)
print()