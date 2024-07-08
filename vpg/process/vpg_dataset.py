import gc
import os.path
import traceback

import dgl
import torch
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.nn.pytorch import MetaPath2Vec
from torch.optim import SparseAdam
from torch.utils.data import DataLoader
from tqdm import tqdm

from prepare.graph import HeteroEdgeType, HomoEdgeType

N_FEATURES = 50


class HeteroMetaDataset(DGLDataset):
    def __init__(self, base, graph_list, force_reload=False, suffix=""):
        self.graphs = []
        self.graph_list = graph_list
        self.base = base
        self.suffix = suffix
        super(HeteroMetaDataset, self).__init__(name=f"HeteroMetaDataset{suffix}",
                                                save_dir=f"/home/jrj/postgraduate/vpg/data/{base}",
                                                verbose=True,
                                                force_reload=force_reload)

    def process(self):
        # if self.suffix == "metapath":
        #     rank = 0
        #     device = torch.device("cuda:{:d}".format(rank))
        #     torch.cuda.set_device(device)
        node_types = ['mapping_var', 'other_state_var', 'local_var', 'function_node']

        # edge_types = [
        #     ('function_node', f"f{HeteroEdgeType.DOMINATOR.value}f", 'function_node'),
        #     ('function_node', f"f{HeteroEdgeType.SUCCESSOR.value}f", 'function_node'),
        #     # ('function_node', HeteroEdgeType.FATHER.value, 'function_node'),
        #     # ('function_node', HeteroEdgeType.SON.value, 'function_node'),
        #     ('function_node', f"f{HeteroEdgeType.CALL.value}f", 'function_node'),
        #     ('function_node', f"f{HeteroEdgeType.CALLED.value}f", 'function_node'),
        #     ('function_node', f"f{HeteroEdgeType.REFTO.value}l", 'local_var'),
        #     ('local_var', f"l{HeteroEdgeType.REFED.value}f", 'function_node')
        # ]
        # for i in range(3):
        #     edge_types.append((node_types[i], f"{node_types[i][0]}{HeteroEdgeType.RF.value}f", 'function_node'))
        #     edge_types.append(('function_node', f"f{HeteroEdgeType.WT.value}{node_types[i][0]}", node_types[i]))
        #     edge_types.append((node_types[i], f"{node_types[i][0]}{HeteroEdgeType.LB.value}f", 'function_node'))
        #     edge_types.append((node_types[i], f"{node_types[i][0]}{HeteroEdgeType.LBN.value}f", 'function_node'))
        # for i in range(3):
        #     for j in range(3):
        #         if not (i == 2 and j == 2):
        #             edge_types.append((node_types[i],
        #                                f"{node_types[i][0]}{HeteroEdgeType.DDF.value}{node_types[j][0]}",
        #                                node_types[j]))
        # for i in range(2):
        #     for j in range(2):
        #         edge_types.append((node_types[i], f"{node_types[i][0]}{HeteroEdgeType.DDC.value}{node_types[j][0]}",
        #                            node_types[j]))
        # else:
        edge_types = [
            ('function_node', HeteroEdgeType.DOMINATOR.value, 'function_node'),
            ('function_node', HeteroEdgeType.SUCCESSOR.value, 'function_node'),
            # ('function_node', HeteroEdgeType.FATHER.value, 'function_node'),
            # ('function_node', HeteroEdgeType.SON.value, 'function_node'),
            ('function_node', HeteroEdgeType.CALL.value, 'function_node'),
            ('function_node', HeteroEdgeType.CALLED.value, 'function_node'),
            ('function_node', HeteroEdgeType.REFTO.value, 'local_var'),
            ('local_var', HeteroEdgeType.REFED.value, 'function_node')
        ]
        for i in range(3):
            edge_types.append((node_types[i], HeteroEdgeType.RF.value, 'function_node'))
            edge_types.append(('function_node', HeteroEdgeType.WT.value, node_types[i]))
            edge_types.append((node_types[i], HeteroEdgeType.LB.value, 'function_node'))
            edge_types.append((node_types[i], HeteroEdgeType.LBN.value, 'function_node'))
        for i in range(3):
            for j in range(3):
                if not (i == 2 and j == 2):
                    edge_types.append((node_types[i], HeteroEdgeType.DDF.value, node_types[j]))
        for i in range(2):
            for j in range(2):
                edge_types.append((node_types[i], HeteroEdgeType.DDC.value, node_types[j]))
        for gi in range(len(self.graph_list)):
            graph = self.graph_list[gi]
            nums_node_dict = dict()
            graph_data = {key: ([], []) for key in edge_types}
            for node_type in node_types:
                num = len(list(filter(lambda x: x[-1] == node_type, graph['nodes_index'])))
                nums_node_dict[node_type] = num
            for edge in graph['edges']:
                edge_type = (edge[0][-1], edge[1], edge[2][-1])
                graph_data[edge_type][0].append(edge[0][0])
                graph_data[edge_type][1].append(edge[2][0])
            g_graph_data = dict()
            for key in edge_types:
                if not (len(graph_data[key][0]) == 0 and len(graph_data[key][1]) == 0):
                    g_graph_data[key] = (torch.tensor(graph_data[key][0]), torch.tensor(graph_data[key][1]))
                else:
                    g_graph_data[key] = ([], [])
            g = dgl.heterograph(g_graph_data, num_nodes_dict=nums_node_dict)
            exact_types = [et for et in g.canonical_etypes if (g.num_edges(et) > 0)]
            for node_type in node_types:
                if nums_node_dict[node_type] == 0: continue
                if node_type == 'mapping_var':
                    metapaths = self.get_metapath(exact_types)
                    tmp_emb = torch.zeros((nums_node_dict[node_type], N_FEATURES))
                    try:
                        for mp in metapaths:
                            model = MetaPath2Vec(g, mp, window_size=2, emb_dim=N_FEATURES)
                            dataloader = DataLoader(torch.arange(1), batch_size=32,
                                                    shuffle=True, collate_fn=model.sample)
                            optimizer = SparseAdam(model.parameters(), lr=0.0025)
                            for (pos_u, pos_v, neg_v) in dataloader:
                                loss = model(pos_u, pos_v, neg_v)
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                            nids = torch.LongTensor(model.local_to_global_nid['mapping_var'])
                            emb = model.node_embed(nids)
                            tmp_emb = torch.add(tmp_emb, emb)
                        avg_emb = tmp_emb / len(metapaths)
                        g.nodes[node_type].data['feats'] = avg_emb
                    except:
                        print(f"{self.suffix}-{gi}")
                        print(traceback.format_exc())
                        return
                    gc.collect()
                else:
                    g.nodes[node_type].data['feats'] = torch.randn(nums_node_dict[node_type], N_FEATURES)
            g.nodes['mapping_var'].data['labels'] = torch.tensor(
                list(dict(sorted(graph['labels'].items())).values()))
            self.graphs.append(g)

    def has_cache(self):
        return os.path.exists(self.graph_list_path)

    def save(self):
        if len(self.graphs) > 0:
            save_graphs(self.graph_list_path, self.graphs)

    def load(self):
        graphs = load_graphs(self.graph_list_path)
        if len(graphs) > 0:
            self.graphs = graphs[0]

    @property
    def graph_list_path(self):
        return os.path.join(self.save_path, "dgl_graph_list.bin")

    def __getitem__(self, item):
        return self.graphs[item]

    def __len__(self):
        return len(self.graphs)

    def __add__(self, other):
        graphs = self.graphs + other.graphs
        result = HeteroMetaDataset(base=self.base, graph_list=[], force_reload=True)
        result.graphs = graphs
        return result

    @staticmethod
    def get_metapath(canonical_etypes):
        possible_metapaths = [
            [('mapping_var', HeteroEdgeType.RF.value, 'function_node'),
             ('function_node', HeteroEdgeType.WT.value, 'mapping_var')],

            [('local_var', HeteroEdgeType.RF.value, 'function_node'),
             ('function_node', HeteroEdgeType.WT.value, 'mapping_var')],

            [('other_state_var', HeteroEdgeType.RF.value, 'function_node'),
             ('function_node', HeteroEdgeType.WT.value, 'mapping_var')],
            # [('mapping_var', HeteroEdgeType.RF.value, 'function_node'),
            #  ('function_node', HeteroEdgeType.SUCCESSOR.value, 'function_node'),
            #  ('function_node', HeteroEdgeType.WT.value, 'mapping_var')],
            #
            # [('mapping_var', HeteroEdgeType.RF.value, 'function_node'),
            #  ('function_node', HeteroEdgeType.DOMINATOR.value, 'function_node'),
            #  ('function_node', HeteroEdgeType.WT.value, 'mapping_var')],
            #
            # [('mapping_var', HeteroEdgeType.RF.value, 'function_node'),
            #  ('function_node', HeteroEdgeType.WT.value, 'local_var'),
            #  ('local_var', HeteroEdgeType.RF.value, 'function_node'),
            #  ('function_node', HeteroEdgeType.WT.value, 'mapping_var')],
            #
            # [('mapping_var', HeteroEdgeType.RF.value, 'function_node'),
            #  ('function_node', HeteroEdgeType.WT.value, 'other_state_var'),
            #  ('other_state_var', HeteroEdgeType.RF.value, 'function_node'),
            #  ('function_node', HeteroEdgeType.WT.value, 'mapping_var')],

            [('local_var', HeteroEdgeType.REFED.value, 'function_node'),
             ('function_node', HeteroEdgeType.DOMINATOR.value, 'function_node')],

            [('function_node', HeteroEdgeType.REFTO.value, 'local_var'),
             ('local_var', HeteroEdgeType.RF.value, 'function_node'),
             ('function_node', HeteroEdgeType.DOMINATOR.value, 'function_node')],

            # [('mapping_var', HeteroEdgeType.LB, 'function_node'),
            #  ('function_node', HeteroEdgeType.DOMINATOR.value, 'function_node')],
            #
            # [('local_var', HeteroEdgeType.LB, 'function_node'),
            #  ('function_node', HeteroEdgeType.DOMINATOR.value, 'function_node')],
            #
            # [('other_state_var', HeteroEdgeType.LB, 'function_node'),
            #  ('function_node', HeteroEdgeType.DOMINATOR.value, 'function_node')],
            #
            # [('mapping_var', HeteroEdgeType.LBN, 'function_node'),
            #  ('function_node', HeteroEdgeType.DOMINATOR.value, 'function_node')],
            #
            # [('function_node', HeteroEdgeType.LBN, 'function_node'),
            #  ('function_node', HeteroEdgeType.DOMINATOR.value, 'function_node')],
            #
            # [('function_node', HeteroEdgeType.CALL, 'function_node'),
            #  ('function_node', HeteroEdgeType.DOMINATOR.value, 'function_node')],
            #
            # [('function_node', HeteroEdgeType.CALLED, 'function_node'),
            #  ('function_node', HeteroEdgeType.DOMINATOR.value, 'function_node')]
        ]
        return list(filter(lambda x: len(set(x).difference(set(canonical_etypes))) == 0, possible_metapaths))


class HeteroVPGDataset(DGLDataset):
    def __init__(self, base, graph_list, suffix=""):
        self.graphs = []
        self.graph_list = graph_list
        self.suffix = suffix
        super(HeteroVPGDataset, self).__init__(name=f"HeteroVPGDataset{suffix}",
                                               save_dir=f"/home/jrj/postgraduate/vpg/data/{base}",
                                               verbose=True)

    def process(self):

        node_types = ['mapping_var', 'other_state_var', 'local_var', 'function_node']
        edge_types = [
            ('function_node', HeteroEdgeType.DOMINATOR.value, 'function_node'),
            ('function_node', HeteroEdgeType.SUCCESSOR.value, 'function_node'),
            # ('function_node', HeteroEdgeType.FATHER.value, 'function_node'),
            # ('function_node', HeteroEdgeType.SON.value, 'function_node'),
            ('function_node', HeteroEdgeType.CALL.value, 'function_node'),
            ('function_node', HeteroEdgeType.CALLED.value, 'function_node'),
            ('function_node', HeteroEdgeType.REFTO.value, 'local_var'),
            ('local_var', HeteroEdgeType.REFED.value, 'function_node')
        ]
        for i in range(3):
            edge_types.append((node_types[i], HeteroEdgeType.RF.value, 'function_node'))
            edge_types.append(('function_node', HeteroEdgeType.WT.value, node_types[i]))
            edge_types.append((node_types[i], HeteroEdgeType.LB.value, 'function_node'))
            edge_types.append((node_types[i], HeteroEdgeType.LBN.value, 'function_node'))
        for i in range(3):
            for j in range(3):
                if not (i == 2 and j == 2):
                    edge_types.append((node_types[i], HeteroEdgeType.DDF.value, node_types[j]))
        for i in range(2):
            for j in range(2):
                edge_types.append((node_types[i], HeteroEdgeType.DDC.value, node_types[j]))

        for graph in tqdm(self.graph_list):
            nums_node_dict = dict()
            graph_data = {key: ([], []) for key in edge_types}
            for node_type in node_types:
                num = len(list(filter(lambda x: x[-1] == node_type, graph['nodes_index'])))
                nums_node_dict[node_type] = num
            for edge in graph['edges']:
                edge_type = (edge[0][-1], edge[1], edge[2][-1])
                graph_data[edge_type][0].append(edge[0][0])
                graph_data[edge_type][1].append(edge[2][0])
            g_graph_data = dict()
            for key in edge_types:
                if not (len(graph_data[key][0]) == 0 and len(graph_data[key][1]) == 0):
                    g_graph_data[key] = (torch.tensor(graph_data[key][0]), torch.tensor(graph_data[key][1]))
                else:
                    g_graph_data[key] = ([], [])
            g = dgl.heterograph(g_graph_data, num_nodes_dict=nums_node_dict)
            for node_type in node_types:
                if nums_node_dict[node_type] == 0: continue
                g.nodes[node_type].data['feats'] = torch.randn(nums_node_dict[node_type], N_FEATURES)
            g.nodes['mapping_var'].data['labels'] = torch.tensor(
                list(dict(sorted(graph['labels'].items())).values()))
            self.graphs.append(g)

    def has_cache(self):
        return os.path.exists(self.graph_list_path)

    def save(self):
        save_graphs(self.graph_list_path, self.graphs)

    def load(self):
        self.graphs = load_graphs(self.graph_list_path)[0]

    @property
    def graph_list_path(self):
        return os.path.join(self.save_path, "dgl_graph_list.bin")

    def __getitem__(self, item):
        return self.graphs[item]

    def __len__(self):
        return len(self.graphs)

    def __add__(self, other: HeteroMetaDataset):
        graphs = self.graphs + other.graphs

        return self


class HomoVPGDataset(DGLDataset):
    def __init__(self, base, graph_list):
        self.graphs = []
        self.graph_list = graph_list
        super().__init__(name="HomoVPGDataset",
                         save_dir=f"/home/jrj/postgraduate/vpg/data/{base}",
                         verbose=True)

    def process(self):
        for graph in tqdm(self.graph_list):
            g = dgl.DGLGraph()
            num_nodes = len(graph['nodes_index'])
            nodes_type = ["" for i in range(num_nodes)]
            for node in graph['nodes_index']:
                nodes_type[node[0]] = node[1]
            nodes_type = torch.tensor(nodes_type)
            g.add_nodes(num_nodes, {'ntype': nodes_type})
            src_list = []
            dst_list = []
            etypes = []
            for edge in graph['edges']:
                edge_type = (edge[0][-1], edge[1], edge[2][-1])
                src_list.append(edge_type[0])
                dst_list.append(edge_type[-1])
                etypes.append(edge_type[1])
                g.add_edges(torch.tensor(src_list), torch.tensor(dst_list), {'etype': torch.tensor(etypes)})
            g.nodes.data['feats'] = torch.randn(num_nodes, N_FEATURES)
            graph['labels'] = dict(sorted(graph['labels'].items()))
            g.nodes[list(graph['labels'].keys())].data['labels'] = torch.tensor(list(graph['labels'].values()))
            self.graphs.append(g)

    def __getitem__(self, item):
        return self.graphs[item], list(self.graph_list[item]['labels'].keys())

    def __len__(self):
        return len(self.graphs)

    def has_cache(self):
        return os.path.exists(self.graph_list_path)

    def save(self):
        save_graphs(self.graph_list_path, self.graphs)

    def load(self):
        self.graphs = load_graphs(self.graph_list_path)[0]

    @property
    def graph_list_path(self):
        return os.path.join(self.save_path, "dgl_graph_list.bin")
