import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import apply_each, edge_type_subgraph
from dgl.nn.pytorch import HeteroGraphConv, GraphConv, GATConv, GATv2Conv, GCN2Conv, HGTConv, SAGEConv
from torch.nn import init


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.gcn_layer = nn.ModuleList()
        self.rel_names = rel_names
        # 实例化HeteroGraphConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregate是聚合函数的类型
        self.gcn_layer.append(HeteroGraphConv({
            rel: GraphConv(in_feats, hid_feats, weight=True)
            for rel in rel_names}, aggregate='mean'))
        for i in range(2):
            self.gcn_layer.append(HeteroGraphConv({
                rel: GraphConv(hid_feats, hid_feats, weight=True)
                for rel in rel_names}, aggregate='mean'))

        self.gcn_layer.append(HeteroGraphConv({
            rel: GraphConv(hid_feats, hid_feats, weight=True)
            for rel in rel_names}, aggregate='mean'))
        self.gcn_layer.append(HeteroGraphConv({
            rel: GraphConv(hid_feats, hid_feats, weight=True)
            for rel in rel_names}, aggregate='mean'))
        # self.conv1 = HeteroGraphConv({
        #     rel: GraphConv(in_feats, hid_feats)
        #     for rel in rel_names}, aggregate='mean')
        # # self.conv2 = HeteroGraphConv({
        # #     rel: GraphConv(hid_feats, hid_feats)
        # #     for rel in rel_names}, aggregate='mean')
        # self.conv3 = HeteroGraphConv({
        #     rel: GraphConv(hid_feats, hid_feats)
        #     for rel in rel_names}, aggregate='mean')
        self.linear = nn.Linear(hid_feats, out_feats)
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, inputs):
        rels = []
        for e in g.canonical_etypes:
            if e[1] in self.rel_names:
                rels.append(e)
        g = edge_type_subgraph(g, rels)
        h = inputs
        for l, layer in enumerate(self.gcn_layer):
            h = layer(g, h)
            if l != len(self.gcn_layer) - 1:
                h = apply_each(h, F.leaky_relu)
                h = apply_each(h, self.dropout)
        # 输入是节点的特征字典
        # h = self.conv1(graph, inputs)
        # h = {k: F.leaky_relu(v) for k, v in h.items()}
        # h = self.conv3(graph, h)
        # h = {k: F.leaky_relu(v) for k, v in h.items()}
        # h = self.conv3(graph, h)
        return self.linear(h["mapping_var"])


class RSAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.sage_layer = nn.ModuleList()
        self.rel_names = rel_names
        self.sage_layer.append(HeteroGraphConv({
            rel: SAGEConv(in_feats, hid_feats, "mean")
            for rel in rel_names}, aggregate='mean'))
        for i in range(4):
            self.sage_layer.append(HeteroGraphConv({
                rel: SAGEConv(hid_feats, hid_feats, "mean")
                for rel in rel_names}, aggregate='mean'))
        self.linear = nn.Linear(hid_feats, out_feats)
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, inputs):
        rels = []
        for e in g.canonical_etypes:
            if e[1] in self.rel_names:
                rels.append(e)
        g = edge_type_subgraph(g, rels)
        h = inputs
        for l, layer in enumerate(self.sage_layer):
            h = layer(g, h)
            if l != len(self.sage_layer) - 1:
                h = apply_each(h, F.leaky_relu)
                h = apply_each(h, self.dropout)
        return self.linear(h["mapping_var"])


class RGAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.rel_names = rel_names
        self.gat_layers.append(
            HeteroGraphConv({
                rel: GATConv(in_feats=in_feats, out_feats=hid_feats, num_heads=heads[0], activation=F.elu)
                for rel in rel_names}, aggregate="mean"
            )
        )
        self.gat_layers.append(
            HeteroGraphConv({
                rel: GATConv(in_feats=hid_feats * heads[0], out_feats=hid_feats, num_heads=heads[1], activation=F.elu)
                for rel in rel_names}, aggregate="mean"
            )
        )
        self.gat_layers.append(
            HeteroGraphConv({
                rel: GATConv(in_feats=hid_feats * heads[1], out_feats=out_feats, num_heads=heads[2], activation=F.elu, )
                for rel in rel_names}, aggregate="mean"
            )
        )

    def forward(self, g, inputs):
        rels = []
        for e in g.canonical_etypes:
            if e[1] in self.rel_names:
                rels.append(e)
        g = edge_type_subgraph(g, rels)
        # 输入是节点的特征字典
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 2:
                for k in h:
                    h[k] = h[k].mean(1)
            else:
                for k in h:
                    h[k] = h[k].flatten(1)
        return h


class RGAT_linear(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, n_heads):
        super().__init__()
        self.layers = nn.ModuleList()
        self.rel_names = rel_names
        self.layers.append(
            HeteroGraphConv(
                {
                    etype: GATConv(in_feats, hid_feats // n_heads, n_heads)
                    for etype in rel_names
                }, aggregate="mean"
            )
        )
        self.layers.append(
            HeteroGraphConv(
                {
                    etype: GATConv(hid_feats, hid_feats // n_heads, n_heads)
                    for etype in rel_names
                }, aggregate="mean"
            )

        )
        self.layers.append(
            HeteroGraphConv(
                {
                    etype: GATConv(hid_feats, hid_feats // n_heads, n_heads)
                    for etype in rel_names
                }, aggregate="mean"
            )

        )
        # self.layers.append(
        #     HeteroGraphConv(
        #         {
        #             etype: GraphConv(hid_feats, hid_feats)
        #             for etype in rel_names
        #         }, aggregate="sum"
        #     )
        #
        # )
        # self.layers.append(
        #     HeteroGraphConv(
        #         {
        #             etype: GATConv(hid_feats, hid_feats // n_heads, n_heads)
        #             for etype in rel_names
        #         }
        #     )
        # )
        # self.layers.append(
        #     HeteroGraphConv(
        #         {
        #             etype: GATConv(hid_feats, hid_feats // n_heads, n_heads)
        #             for etype in rel_names
        #         }
        #     )
        # )
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hid_feats, out_feats)

    def forward(self, g, inputs):
        rels = []
        for e in g.canonical_etypes:
            if e[1] in self.rel_names:
                rels.append(e)
        g = edge_type_subgraph(g, rels)
        # 输入是节点的特征字典
        h = inputs
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            h = apply_each(
                h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2])
            )
            if l != len(self.layers) - 1:
                h = apply_each(h, F.leaky_relu)
                h = apply_each(h, self.dropout)
        return self.linear(h["mapping_var"])


class RGAT_GCN_concat(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, n_heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.gcn_layers = nn.ModuleList()
        self.gat_layers.append(
            HeteroGraphConv(
                {
                    etype: GATConv(in_feats, (hid_feats // 2) // n_heads, n_heads)
                    for etype in rel_names
                }, aggregate="mean"
            )
        )
        self.gat_layers.append(
            HeteroGraphConv(
                {
                    etype: GATConv((hid_feats // 2), (hid_feats // 2) // n_heads, n_heads)
                    for etype in rel_names
                }, aggregate="mean"
            )
        )
        self.gcn_layers.append(
            HeteroGraphConv(
                {
                    etype: GraphConv(in_feats, (hid_feats // 2))
                    for etype in rel_names
                }, aggregate="mean"
            )

        )
        self.gcn_layers.append(
            HeteroGraphConv(
                {
                    etype: GraphConv((hid_feats // 2), (hid_feats // 2))
                    for etype in rel_names
                }, aggregate="mean"
            )

        )
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(hid_feats, out_feats)

    def forward(self, g, inputs):
        # 输入是节点的特征字典
        gat_h = inputs
        gcn_h = inputs
        for l, layer in enumerate(self.gat_layers):
            gat_h = layer(g, gat_h)
            gat_h = apply_each(
                gat_h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2])
            )
            if l != len(self.gat_layers) - 1:
                gat_h = apply_each(gat_h, F.leaky_relu)
                gat_h = apply_each(gat_h, self.dropout)
        for l, layer in enumerate(self.gcn_layers):
            gcn_h = layer(g, gcn_h)
            if l != len(self.gat_layers) - 1:
                gcn_h = apply_each(gcn_h, F.leaky_relu)
                gcn_h = apply_each(gcn_h, self.dropout)

        h = dict()
        for item in inputs:
            h[item] = torch.cat((gat_h[item], gcn_h[item]), dim=1)
        return self.linear(h["mapping_var"])


class RGCN_GAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, gcn_layers, n_heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.gcn_layers = nn.ModuleList()
        self.rel_names = rel_names
        self.gcn_layers.append(
            HeteroGraphConv(
                {
                    etype: GraphConv(in_feats, hid_feats)
                    for etype in rel_names
                }, aggregate="mean"
            )

        )
        for i in range(gcn_layers - 1):
            self.gcn_layers.append(
                HeteroGraphConv(
                    {
                        etype: GraphConv(hid_feats, hid_feats)
                        for etype in rel_names
                    }, aggregate="mean"
                )

            )
        self.gat_layers.append(
            HeteroGraphConv(
                {
                    etype: GATConv(hid_feats, hid_feats // n_heads, n_heads)
                    for etype in rel_names
                }, aggregate="mean"
            )
        )
        # self.gat_layers.append(
        #     HeteroGraphConv(
        #         {
        #             etype: GATConv(hid_feats, hid_feats // n_heads, n_heads)
        #             for etype in rel_names
        #         }, aggregate="mean"
        #     )
        # )

        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear((hid_feats // n_heads) * n_heads, out_feats)

    def forward(self, g, inputs):
        rels = []
        for e in g.canonical_etypes:
            if e[1] in self.rel_names:
                rels.append(e)
        g = edge_type_subgraph(g, rels)
        # 输入是节点的特征字典
        h = inputs
        for l, layer in enumerate(self.gcn_layers):
            h = layer(g, h)
            h = apply_each(h, F.leaky_relu)
            h = apply_each(h, self.dropout)
        for l, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            h = apply_each(
                h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2])
            )
            if l != len(self.gat_layers) - 1:
                h = apply_each(h, F.leaky_relu)
                h = apply_each(h, self.dropout)

        return self.linear(h["mapping_var"])


class RSAGE_GAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, n_heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.sage_layers = nn.ModuleList()
        self.rel_names = rel_names
        self.sage_layers.append(HeteroGraphConv({
            rel: SAGEConv(in_feats, hid_feats, "mean")
            for rel in rel_names}, aggregate='mean')
        )
        for i in range(3):
            self.sage_layers.append(HeteroGraphConv({
                rel: SAGEConv(hid_feats, hid_feats, "mean")
                for rel in rel_names}, aggregate='mean'))
        self.gat_layers.append(
            HeteroGraphConv(
                {
                    etype: GATConv(hid_feats, hid_feats // n_heads, n_heads)
                    for etype in rel_names
                }, aggregate="mean"
            )
        )
        # self.gat_layers.append(
        #     HeteroGraphConv(
        #         {
        #             etype: GATConv(hid_feats, hid_feats // n_heads, n_heads)
        #             for etype in rel_names
        #         }, aggregate="mean"
        #     )
        # )
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hid_feats, out_feats)

    def forward(self, g, inputs):
        rels = []
        for e in g.canonical_etypes:
            if e[1] in self.rel_names:
                rels.append(e)
        g = edge_type_subgraph(g, rels)
        # 输入是节点的特征字典
        h = inputs
        for l, layer in enumerate(self.sage_layers):
            h = layer(g, h)
            h = apply_each(h, F.leaky_relu)
            h = apply_each(h, self.dropout)
        for l, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            h = apply_each(
                h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2])
            )
            if l != len(self.gat_layers) - 1:
                h = apply_each(h, F.leaky_relu)
                h = apply_each(h, self.dropout)

        return self.linear(h["mapping_var"])


class HGT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = HeteroGraphConv({
            rel: GCN2Conv(in_feats, layer=1, alpha=0.5, project_initial_features=True, allow_zero_in_degree=True,
                          activation=F.leaky_relu)
            for rel in rel_names}, aggregate='mean')
        self.conv2 = HeteroGraphConv({
            rel: GCN2Conv(in_feats, layer=2, alpha=0.5, project_initial_features=True, allow_zero_in_degree=True)
            for rel in rel_names}, aggregate='mean')
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        h = inputs
        h = self.conv1(g, h, feat_0=inputs)
        h = apply_each(h, self.dropout)
        h = self.conv2(g, h, feat_0=inputs)
        return self.linear(h["mapping_var"])





