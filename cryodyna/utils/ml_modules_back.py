from typing import Union, List
import numpy as np
import torch
from torch import nn
# from torch_geometric.nn import GATConv
# from utils.layers import Linear  # Why this?
from torch.nn import Linear
# from cryostar.openfold.utils.tensor_utils import one_hot
from torch_geometric.nn import MessagePassing, knn_graph, TransformerConv
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean
import torch.nn.functional as F

class ResLinear(nn.Module):

    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.linear = Linear(in_chs, out_chs)

    def forward(self, x):
        return self.linear(x) + x


class MLP(nn.Module):

    def __init__(self, in_dims: List[int], out_dims: List[int]):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims

        layers = []
        for (i, o) in zip(in_dims, out_dims):
            layers.append(ResLinear(i, o) if i == o else Linear(i, o))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LayerNorm(o))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class Linear_w_crossattention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Sequential(Linear(in_dim,out_dim),nn.ELU(inplace=True),nn.LayerNorm(out_dim))
        self.attention = TransformerConv(out_dim,out_dim,heads=4,concat=False)

    def forward(self, x):
        x = self.linear(x)
        img_img_idx= complete_graph(x.shape[0],x.device)
        x = self.attention(x,img_img_idx)
        return x


class Encoder(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: Union[int, List[int]], out_dim: int, num_hidden_layers=3):
        super().__init__()
        self.in_dim = in_dim
        if isinstance(hidden_dim, int):
            self.hidden_dim = (hidden_dim, ) * num_hidden_layers
        elif isinstance(hidden_dim, (list, tuple)):
            assert len(hidden_dim) == num_hidden_layers
            self.hidden_dim = hidden_dim
        else:
            raise NotImplementedError
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers

        self.input_layer = nn.Sequential(
            ResLinear(in_dim, self.hidden_dim[0]) if in_dim == self.hidden_dim[0] else Linear(
                in_dim, self.hidden_dim[0]), nn.ELU(inplace=True))
        self.mlp = MLP(self.hidden_dim[:-1], self.hidden_dim[1:])

        self.output_layer = Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.output_layer(self.mlp(self.input_layer(x)))


def complete_graph(num_nodes, device):
    idx = torch.arange(num_nodes, device=device)
    row = idx.repeat_interleave(num_nodes)
    col = idx.repeat(num_nodes)
    return torch.stack([row, col], dim=0)

class EncoderTransformerConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, latent_dim, conv_heads=1, dropout=0.0):
        """
        input_dim: 每张图像输入的特征维度
        hidden_dim: 三层 MLP 中间层维度
        embed_dim: 每张图像编码后的 embedding 维度
        latent_dim: VAE 隐变量维度（输出 mu 和 logvar 的维度）
        conv_heads: TransformerConv 的头数
        dropout: dropout 概率
        """
        super(EncoderTransformerConv, self).__init__()
        
        # 三层 MLP 对每张图像进行单独编码
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3 = nn.Linear(hidden_dim, embed_dim)
        
        # TransformerConv 融合图像间信息
        self.transformer_conv = TransformerConv(embed_dim, embed_dim, heads=conv_heads,
                                                  dropout=dropout, add_self_loops=False)
        
        # 两个不同的线性变换，用于自我表示和融合表示
        self.W_self = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_context = nn.Linear(embed_dim * conv_heads, embed_dim, bias=False)
        
        # 两层 MLP 输出 VAE bottleneck 参数（均值和 logvar）
        self.fc1 = nn.Linear(2 * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim * 2)  # 输出: [mu, logvar]
    
    def forward(self, x):
        """
        x: [N, input_dim]，其中 N 是图像数量
        """
        N = x.size(0)
        device = x.device
        
        # 1. 单图编码：经过三层 MLP 得到 H: [N, embed_dim]
        h = F.relu(self.mlp1(x))
        h = F.relu(self.mlp2(h))
        H = F.relu(self.mlp3(h))
        
        # 2. 自我表示：h_self = W_self(H)
        h_self = self.W_self(H)  # [N, embed_dim]
        
        # 3. 图像间信息融合
        # 构造完整图（不含自环）
        edge_index = complete_graph(N, device)  # [2, N*(N-1)]
        # TransformerConv 融合信息，输出形状为 [N, embed_dim * conv_heads]
        H_fuse = self.transformer_conv(H, edge_index)
        # 将融合表示降维：h_fuse = W_context(H_fuse) → [N, embed_dim]
        h_fuse = self.W_context(H_fuse)
        
        # 4. 组合表示：拼接 h_self 和 h_fuse → [N, 2 * embed_dim]
        combined = torch.cat([h_self, h_fuse], dim=-1)
        
        # 5. 经过两层 MLP 输出 VAE bottleneck 参数
        hidden = F.relu(self.fc1(combined))
        bottleneck = self.fc2(hidden)  # [N, latent_dim*2]
        mu, logvar = torch.chunk(bottleneck, 2, dim=-1)
        
        return mu, logvar
    
class VAEEncoder(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: Union[int, List[int]], out_dim: int, num_hidden_layers=3):
        super().__init__()
        self.in_dim = in_dim
        if isinstance(hidden_dim, int):
            self.hidden_dim = (hidden_dim, ) * num_hidden_layers
        elif isinstance(hidden_dim, (list, tuple)):
            assert len(hidden_dim) == num_hidden_layers
            self.hidden_dim = hidden_dim
        else:
            raise NotImplementedError
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers

        self.input_layer = nn.Sequential(
            ResLinear(in_dim, self.hidden_dim[0]) if in_dim == self.hidden_dim[0] else Linear(
                in_dim, self.hidden_dim[0]), nn.ELU(inplace=True))
        self.mlp = MLP(self.hidden_dim[:-1], self.hidden_dim[1:])

        self.mean_layer = Linear(self.hidden_dim[-1], out_dim)
        # self.var_layer = Linear(self.hidden_dim[-1], out_dim)

    def forward(self, x):
        x = self.mlp(self.input_layer(x))
        mean = self.mean_layer(x)
        return mean
        # log_var = self.var_layer(x)
        # return mean, log_var

# class GATEncoder(nn.Module):

#     def __init__(self, in_dim: int, hidden_dim: Union[int, List[int]], attention_layer:int ,out_dim: int, num_hidden_layers=3):
#         super().__init__()
#         self.in_dim = in_dim
#         if isinstance(hidden_dim, int):
#             self.hidden_dim = (hidden_dim, ) * num_hidden_layers
#         elif isinstance(hidden_dim, (list, tuple)):
#             assert len(hidden_dim) == num_hidden_layers
#             self.hidden_dim = hidden_dim
#         else:
#             raise NotImplementedError
#         self.out_dim = out_dim
#         self.num_hidden_layers = num_hidden_layers

#         self.input_layer = nn.Sequential(
#             ResLinear(in_dim, self.hidden_dim[0]) if in_dim == self.hidden_dim[0] else Linear(
#                 in_dim, self.hidden_dim[0]), nn.ReLU(inplace=True))
#         self.mlp1 = MLP(self.hidden_dim[:attention_layer], self.hidden_dim[1:attention_layer+1])
#         self.attention_layer = TransformerConv(hidden_dim[attention_layer],hidden_dim[attention_layer],heads=4,concat=False)
#         # self.mlp2 = MLP([self.hidden_dim[attention_layer] * 2], [self.hidden_dim[attention_layer]])
#         # self.mlp2 = MLP(self.hidden_dim[attention_layer:-1] , self.hidden_dim[attention_layer+1:])
#         self.mean_layer = Linear(self.hidden_dim[-1], out_dim)
#         # self.var_layer = Linear(self.hidden_dim[-1], out_dim)

#     def forward(self, x):
#         x = self.mlp1(self.input_layer(x))
#         img_img_idx= complete_graph(x.shape[0],x.device)
#         x = self.attention_layer(x,img_img_idx)
#         # x = self.mlp2(x)
#         # x = self.mlp3(x)
#         mean = self.mean_layer(x)
#         # log_var = self.var_layer(x)
#         # 
#         return mean
class MultiScaleGATEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: Union[int, List[int]], attention_layer:int ,out_dim: int, num_hidden_layers=3):
        super().__init__()
        self.in_dim = in_dim
        if isinstance(hidden_dim, int):
            self.hidden_dim = (hidden_dim, ) * num_hidden_layers
        elif isinstance(hidden_dim, (list, tuple)):
            assert len(hidden_dim) == num_hidden_layers
            self.hidden_dim = hidden_dim
        else:
            raise NotImplementedError
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.mlp1 = MLP(np.concatenate(([in_dim],self.hidden_dim[:attention_layer])), self.hidden_dim[:attention_layer+1])
        self.attn_layers = [TransformerConv(i,o,heads=4,concat=False) for i,o in zip(hidden_dim[attention_layer:-1],hidden_dim[attention_layer+1:])]
        self.pre_norm = [nn.LayerNorm(o) for i,o in zip(hidden_dim[attention_layer:-1],hidden_dim[attention_layer+1:])]
        self.ff_layers = [nn.Sequential(nn.Linear(i,o),nn.ELU(inplace=True)) for i,o in zip(hidden_dim[attention_layer:-1],hidden_dim[attention_layer+1:])]
        self.post_norm = [nn.LayerNorm(o) for i,o in zip(hidden_dim[attention_layer:-1],hidden_dim[attention_layer+1:])]
        # self.mlp2 = nn.Sequential(*layers)
        # self.input_layer = nn.Sequential(
        #     ResLinear(in_dim, self.hidden_dim[0]) if in_dim == self.hidden_dim[0] else Linear(
        #         in_dim, self.hidden_dim[0]), nn.ReLU(inplace=True),nn.LayerNorm(self.hidden_dim[0]))
        # self.mlp1 = MLP(self.hidden_dim[:attention_layer], self.hidden_dim[1:attention_layer+1])
        # self.attention_layer = TransformerConv(hidden_dim[attention_layer],hidden_dim[attention_layer],heads=4,concat=False)
        # # self.mlp2 = MLP([self.hidden_dim[attention_layer] * 2], [self.hidden_dim[attention_layer]])
        # self.mlp2 = MLP(self.hidden_dim[attention_layer:-1] , self.hidden_dim[attention_layer+1:])
        # self.mean_layer = Linear(self.hidden_dim[-1], out_dim)
        # self.var_layer = Linear(self.hidden_dim[-1], out_dim)

    def forward(self, x):
        x = self.mlp1(x)
        img_img_idx= complete_graph(x.shape[0],x.device)
        # import pdb;pdb.set_trace()
        for it, attn_layer in enumerate(self.attn_layers):
            attn_layer = attn_layer.to(x.device)
            pre_norm = self.pre_norm[it].to(x.device)
            post_norm = self.post_norm[it].to(x.device)
            x = x + attn_layer(x,img_img_idx)
            x = pre_norm(x)
            ff_layer = self.ff_layers[it].to(x.device)
            x = x + ff_layer(x)
            x = post_norm(x)
        # x = self.encoder(x)
        # mean = self.mean_layer(x)
        # log_var = self.var_layer(x)
        return x
    
class GATEncoder(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: Union[int, List[int]], attention_layer:int ,out_dim: int, num_hidden_layers=3):
        super().__init__()
        self.in_dim = in_dim
        if isinstance(hidden_dim, int):
            self.hidden_dim = (hidden_dim, ) * num_hidden_layers
        elif isinstance(hidden_dim, (list, tuple)):
            assert len(hidden_dim) == num_hidden_layers
            self.hidden_dim = hidden_dim
        else:
            raise NotImplementedError
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers

        self.input_layer = nn.Sequential(
            ResLinear(in_dim, self.hidden_dim[0]) if in_dim == self.hidden_dim[0] else Linear(
                in_dim, self.hidden_dim[0]), nn.ELU(inplace=True))
        self.mlp1 = MLP(self.hidden_dim[:attention_layer], self.hidden_dim[1:attention_layer+1])
        # self.norm1 = nn.LayerNorm(self.hidden_dim[attention_layer])
        self.attention_layer = TransformerConv(hidden_dim[attention_layer],hidden_dim[attention_layer],heads=4,concat=False)
        self.dropout = nn.Dropout(p=0.5)
        self.norm1 = nn.LayerNorm(self.hidden_dim[attention_layer])
        # self.mlp2 = MLP([self.hidden_dim[attention_layer] * 2], [self.hidden_dim[attention_layer]])
        self.mlp2 = MLP(self.hidden_dim[attention_layer:-1] , self.hidden_dim[attention_layer+1:])
        self.mean_layer = Linear(self.hidden_dim[-1], out_dim)
        # self.var_layer = Linear(self.hidden_dim[-1], out_dim)

    def forward(self, x):
        x = self.mlp1(self.input_layer(x))
        # x = self.all_gather(x)
        img_img_idx= complete_graph(x.shape[0],x.device)
        # x = self.norm1(x)
        x = self.attention_layer(x,img_img_idx)
        x = self.dropout(x)
        x = self.norm1(x)
        x = self.mlp2(x)
        # x = self.mlp3(x)
        mean = self.mean_layer(x)
        # log_var = self.var_layer(x)
        return mean


class Decoder(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: Union[int, List[int]], out_dim: int, num_hidden_layers=3):
        super().__init__()
        self.in_dim = in_dim
        if isinstance(hidden_dim, int):
            self.hidden_dim = (hidden_dim, ) * num_hidden_layers
        elif isinstance(hidden_dim, (list, tuple)):
            assert len(hidden_dim) == num_hidden_layers
            self.hidden_dim = hidden_dim
        else:
            raise NotImplementedError
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers

        self.input_layer = nn.Sequential(
            ResLinear(in_dim, self.hidden_dim[0]) if in_dim == self.hidden_dim[0] else Linear(
                in_dim, self.hidden_dim[0]) ,nn.ELU(inplace=True))
        self.mlp = MLP(self.hidden_dim[:-1], self.hidden_dim[1:])

        self.out_layer = Linear(self.hidden_dim[-1], out_dim)

    def forward(self, x):
        x = self.mlp(self.input_layer(x))
        return self.out_layer(x)


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1
        edge_attr_nf = 1
        # edge_bond_nf = 1    
        # self.dist_range = np.append(np.arange(2.3125,22,0.3125),30)
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edge_attr_nf , hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        self.gate_mlp = nn.Sequential(nn.Linear(2 * hidden_nf, hidden_nf), act_fn, nn.Linear(hidden_nf, 64), nn.Sigmoid())
        self.h_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 3))
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        # gate = self.gate_mlp(torch.cat([source, target], dim=1))
        # edge_attr = torch.mean(edge_attr * gate * torch.from_numpy(self.dist_range[None,:]).to(gate.device).to(target.dtype),axis=-1).unsqueeze(1)
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            # import pdb;pdb.set_trace()
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index[:,0] , edge_index[:,1]
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat,hidden):
        row, col = edge_index[:,0] , edge_index[:,1]    
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord += 0.2 * agg
        # import pdb;pdb.set_trace()
        coord += self.h_mlp(hidden)
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index[:,0] , edge_index[:,1]
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index[:,0] , edge_index[:,1]
        edge_attr = edge_attr.to(h.dtype)
        # edge_bond = edge_bond.to(h.dtype)
        radial, coord_diff = self.coord2radial(edge_index, coord)
        # edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, edge_bond)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat,h)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNNDecoder(nn.Module):
    def __init__(self, hidden_nf, n_layers=3, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, tanh=False):
    # def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=3, residual=True, attention=False, normalize=False, tanh=False):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNNDecoder, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(self.hidden_nf, self.hidden_nf)
        # self.embedding_in_1 = nn.Linear(in_node_nf, self.hidden_nf)
        # self.embedding_in_2 = nn.Linear(self.hidden_nf, self.hidden_nf)
        # self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
    def forward(self, h, x, edges, edge_attr):
        # h = self.embedding_in_1(h)
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        # h = self.embedding_out(h)
        return x

class EGNNDecoder_BB(nn.Module):
    def __init__(self, hidden_nf, n_layers=3, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, tanh=False):
    # def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=3, residual=True, attention=False, normalize=False, tanh=False):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNNDecoder_BB, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(self.hidden_nf, self.hidden_nf)
        # self.embedding_in_1 = nn.Linear(in_node_nf, self.hidden_nf)
        # self.embedding_in_2 = nn.Linear(self.hidden_nf, self.hidden_nf)
        self.embedding_out = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),act_fn,nn.Linear(self.hidden_nf,6))
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
    def forward(self, h, x, edges, edge_attr):
        # h = self.embedding_in_1(h)
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        # h = self.embedding_out(h)
        affines = self.embedding_out(h)
        return affines
        # quaternions = self.embedding_out(h)
        # return x , quaternions
    
def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

    

def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std

# class EdgeScorer(nn.Module):
#     def __init__(self, node_dim):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(node_dim * 2 + 2, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1)
#         )

#     def forward(self, z_i, z_j, dist_ij, plasticity_ij):
#         feat = torch.cat([z_i, z_j, dist_ij, plasticity_ij], dim=-1)
#         score = self.mlp(feat)
#         return score  # [E, 1]
    
# def build_scored_topk_edges(z, dist, pae, top_k=32, scorer=None):
#     B, L, D = z.shape
#     z_i = z.unsqueeze(2).expand(B, L, L, D)
#     z_j = z.unsqueeze(1).expand(B, L, L, D)
#     dist = dist.unsqueeze(0).eunsqueeze(-1).expand(B, L, L, 1)
#     pae = pae.unsqueeze(0).unsqueeze(-1).expand(B, L, L, 1)
#     edge_feat = torch.cat([z_i, z_j, dist, pae], dim=-1).view(B * L * L, -1)
#     with torch.no_grad():
#         scores = scorer(edge_feat)
#         scores = scores.view(B, L, L)
#     scores = scores.masked_fill(torch.eye(L).bool().unsqueeze(0), -float('inf'))
#     topk_scores, topk_indices = torch.topk(scores, k=top_k, dim=-1)

#     row = torch.arange(L, device=device).view(1, L, 1).expand(B, L, top_k)
#     batch_ids = torch.arange(B, device=device).view(B, 1, 1).expand(B, L, top_k)
#     col = topk_indices

#     row = row.reshape(-1)
#     col = col.reshape(-1)
#     batch_ids = batch_ids.reshape(-1)
#     edge_index = torch.stack([
#         batch_ids * L + row,
#         batch_ids * L + col
#     ], dim=0)

#     zi = z[batch_ids, row]
#     zj = z[batch_ids, col]
#     xi = coords[row]
#     xj = coords[col]
#     dist = torch.norm(xi - xj, dim=-1, keepdim=True)
#     plast = torch.zeros_like(dist) if plasticity is None else plasticity[row, col].unsqueeze(-1)
#     edge_attr = torch.cat([zi, zj, dist, plast], dim=-1)

#     return edge_index, edge_attr

# class EdgeMLPConv(nn.Module):
#     def __init__(self, in_dim, edge_dim, out_dim):
#         super().__init__()
#         self.msg_mlp = nn.Sequential(
#             nn.Linear(2 * in_dim + edge_dim, 16),
#             nn.ReLU(),
#             nn.Linear(16, out_dim)
#         )
#         self.norm = nn.LayerNorm(out_dim)

#     def forward(self, x, edge_index, edge_attr):
#         src, tgt = edge_index
#         x_i, x_j = x[tgt], x[src]
#         msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
#         messages = self.msg_mlp(msg_input)
#         out = scatter_mean(messages, tgt, dim=0, dim_size=x.size(0))
#         return self.norm(out + x)

# class EdgeScoredGraphDecoder(nn.Module):
#     def __init__(self,  hidden_dim, edge_dim, edge_pae, edge_dist ,top_k=32):
#         super().__init__()
#         self.top_k = top_k
#         self.edge_scorer = nn.Sequential(
#             nn.Linear(hidden_dim * 2 + 2, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1)
#         )
#         self.conv = EdgeMLPConv(hidden_dim, hidden_dim, hidden_dim)
#         self.edge_pae = edge_pae
#         self.edge_dist = edge_dist
#     def forward(self, z, coords):
#         B, L, D = z.shape
#         edge_index, edge_attr = build_scored_topk_edges(z, self.edge_dist, self.edge_pae, self.top_k, self.edge_scorer)
#         z_flat = z.reshape(B * L, D)
#         z_out = self.conv(z_flat, edge_index, edge_attr)
#         return z_out.view(B, L, -1)
    
# class DualGATBlock(nn.Module):
#     def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
#         super().__init__()
#         self.num_heads = num_heads
#         self.cross_gat = GATConv(in_dim, in_dim // num_heads, heads=num_heads, concat=True, dropout=dropout)
#         self.intra_gat = GATConv(in_dim, out_dim // num_heads, heads=num_heads, concat=True, dropout=dropout)
#         self.norm1 = nn.LayerNorm(in_dim)
#         self.norm2 = nn.LayerNorm(out_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.in_dim = in_dim
#         self.out_dim = out_dim

#     def build_cross_sample_edge_index(self, B, L, device):
#         rows, cols = [], []
#         for l in range(L):
#             node_ids = torch.arange(B, device=device) * L + l
#             row = node_ids.repeat_interleave(B)
#             col = node_ids.repeat(B)
#             mask = row != col
#             rows.append(row[mask])
#             cols.append(col[mask])
#         return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)

#     def build_intra_sample_edge_index(self, B, L, device):
#         rows, cols = [], []
#         base = torch.arange(B, device=device) * L
#         for b in base:
#             node_ids = b + torch.arange(L, device=device)
#             row = node_ids.repeat_interleave(L)
#             col = node_ids.repeat(L)
#             mask = row != col
#             rows.append(row[mask])
#             cols.append(col[mask])
#         return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)

#     def forward(self, z):
#         B, L, C = z.shape
#         x = z.reshape(B * L, C)
#         device = x.device

#         edge_index_cross = self.build_cross_sample_edge_index(B, L, device)
#         edge_index_intra = self.build_intra_sample_edge_index(B, L, device)

#         x1 = self.cross_gat(x, edge_index_cross)           # [B*L, C]
#         x1 = self.norm1(x1 + x)                            # Residual + Norm

#         x2 = self.intra_gat(x1, edge_index_intra)          # [B*L, out_dim]
#         x2 = self.norm2(x2)                                # Final LayerNorm

#         return x2.reshape(B, L, self.out_dim)
    
# class DualGAT(nn.Module):
    
#     def __init__(self, in_dim, hidden_dim, vn_num, points_num = 64 , num_heads=4, num_layers=2, dropout=0.1):
#         super().__init__()
#         self.z_to_hidden = nn.Linear(in_dim, vn_num * hidden_dim)
#         layers = []
#         for i in range(num_layers):
#             layers.append(DualGATBlock(
#                 in_dim=in_dim if i == 0 else hidden_dim,
#                 out_dim=hidden_dim,
#                 num_heads=num_heads,
#                 dropout=dropout
#             ))
#         self.layers = nn.ModuleList(layers)
#         self.h_to_coord = nn.Linear(hidden_dim, 3)
    
#     def forward(self, z):
#         h = self.z_to_hidden(z)
#         for layer in self.layers:
#             h = layer(h)
#         coords = self.h_to_coord(h)
#         return coords

class CrossGraphAttention(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=heads, batch_first=True)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, batch):
        # x: (N_total, in_dim), batch: (N_total,)
        # 假设每个图都有 L 个节点，按位置对齐
        x_dense, mask = to_dense_batch(x, batch)  # (B, L, in_dim)
        x_attn, _ = self.attn(x_dense, x_dense, x_dense, key_padding_mask=~mask)
        x_out = self.linear(x_attn)
        x_out = x_out[mask]  # (N_total, out_dim)
        return x_out
    
class DistanceAwareMPNN(MessagePassing):
    def __init__(self, node_dim, edge_dim=1, coord_dim=3, k=32):
        super().__init__(aggr='mean')
        self.k = k
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim + 1, 16),
            nn.ReLU(),
            nn.Linear(16, edge_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, 16),
            nn.ReLU(),
            nn.Linear(16, node_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(edge_dim, 16),
            nn.ReLU(),
            nn.Linear(16, coord_dim),
        )

    def forward(self, x, pos, edge_attr, batch, node_idx):
        # 重新构建 KNN 图（动态）
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=False)
        row, col = edge_index
        src_local = node_idx[row]  # [E]
        tgt_local = node_idx[col]  # [E]
        edge_attr = edge_attr[src_local, tgt_local]  # [E, d_e]
        return self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)

    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        rel_pos = pos_j - pos_i
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        # import pdb;pdb.set_trace()
        edge_input = torch.cat([x_i, x_j, edge_attr.unsqueeze(-1), dist], dim=-1)
        e_ij = self.edge_mlp(edge_input)
        return torch.cat([e_ij, rel_pos], dim=-1)

    def update(self, aggr_out, x, pos):
        edge_feat, rel_pos = aggr_out.split([aggr_out.shape[1] - 3, 3], dim=-1)
        x_new = self.node_mlp(torch.cat([x, edge_feat], dim=-1))
        delta_pos = self.coord_mlp(edge_feat)
        pos_new = pos + delta_pos
        return x_new, pos_new

class DualGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, points_num, edge_dim=1, num_layers=2):
        super().__init__()
        self.points_num = points_num
        self.hidden_dim = hidden_dim
        self.z_to_hidden = nn.Linear(in_dim, points_num * hidden_dim)
        self.cross_attn = CrossGraphAttention(hidden_dim, hidden_dim)
        self.mpnn_layers = nn.ModuleList([
            DistanceAwareMPNN(hidden_dim, edge_dim) for _ in range(num_layers)
        ])
        # self.coord_predictors = nn.ModuleList([
        #     nn.Linear(hidden_dim, 3) for _ in range(num_layers)
        # ])

    def forward(self, x, pos, edge_attr):
        N = x.shape[0]
        h = self.z_to_hidden(x)
        edge_attr = edge_attr
        h = h.reshape(N * self.points_num, self.hidden_dim)
        batch = torch.arange(N).repeat_interleave(self.points_num).to(h.device)
        node_idx =  torch.arange(self.points_num).unsqueeze(0).repeat(N, 1).flatten().to(h.device)
        h = self.cross_attn(h, batch)
        coord_preds = []
        pos = pos.unsqueeze(0).repeat(N,1,1).reshape(N * self.points_num,-1).to(h.device)
        edge_attr = edge_attr.to(h.device)
        for mpnn in self.mpnn_layers:
            h, pos = mpnn(h, pos, edge_attr, batch,node_idx)
            # coord_pred = coord_head(x)
            coord_preds.append(pos)
        return coord_preds


import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv, GATConv

class ResidueIntraBlockGNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gnn = GCNConv(dim, dim)

    def forward(self, x, edge_index, sec_ids):
        # 保留同一结构块内的边
        row, col = edge_index
        mask = sec_ids[row] == sec_ids[col]
        filtered_edge_index = edge_index[:, mask]
        return self.gnn(x, filtered_edge_index)

class SSEPooling(nn.Module):
    def forward(self, x, sec_ids):
        # x: (N_residues, C), sec_ids: (N_residues,)
        return scatter_mean(x, sec_ids, dim=0)

class MetaGraphGNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gnn = GATConv(dim, dim, heads=4, concat=False)

    def forward(self, meta_x, meta_edge_index):
        return self.gnn(meta_x, meta_edge_index)

class MetaToResidueFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, res_x, meta_x, sec_ids):
        # meta_x: (N_meta, C) → broadcast
        meta_feat = meta_x[sec_ids]  # (N_residues, C)
        return self.fuse(torch.cat([res_x, meta_feat], dim=-1))

class HierarchicalProteinGNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.intra_ss_gnn = ResidueIntraBlockGNN(dim)
        self.pooling = SSEPooling()
        self.meta_gnn = MetaGraphGNN(dim)
        self.fusion = MetaToResidueFusion(dim)

    def forward(self, x, edge_index, sec_ids, meta_edge_index):
        # [1] 残基内部更新
        x_updated = self.intra_ss_gnn(x, edge_index, sec_ids)

        # [2] Pool 到二级结构级别
        meta_x = self.pooling(x_updated, sec_ids)

        # [3] 二级结构间传播
        meta_x_updated = self.meta_gnn(meta_x, meta_edge_index)

        # [4] 注入回残基表示
        out = self.fusion(x_updated, meta_x_updated, sec_ids)

        return out  # shape: (N_residues, dim)

from torch_scatter import scatter_mean
from torch_geometric.nn import GATv2Conv

def expand_batch(batch_size, node_idx, meta_edge_idx, edge_dist, meta_2_node_edge, meta_2_node_vector):
    batch_node_idx = []
    batch_meta_edge_idx = []
    batch_meta_2_node_idx = []
    batch_meta_2_node_vector = []
    for i in range(batch_size):
        shifted_idx = node_idx + i * (torch.max(node_idx)+1)
        edge_idx_shifted = meta_edge_idx + i * (torch.max(node_idx)+1)
        batch_node_idx.append(shifted_idx)
        batch_meta_edge_idx.append(edge_idx_shifted)
        node_offset = i * (torch.max(meta_2_node_edge[1])+1)
        meta_offset = i * (torch.max(node_idx)+1)   # <-- global metanode offset
        ei = meta_2_node_edge.clone()
        ei[0] += meta_offset  # metanode idx
        ei[1] += node_offset  # node idx
        batch_meta_2_node_idx.append(ei)
        batch_meta_2_node_vector.append(meta_2_node_vector)
    batch_node_idx = torch.cat(batch_node_idx)
    batch_meta_edge_idx = torch.cat(batch_meta_edge_idx, dim=1)
    batch_edge_dist = edge_dist.repeat(batch_size)
    batch_meta_2_node_idx = torch.cat(batch_meta_2_node_idx, dim=1)  # [2, E_total]
    batch_meta_2_node_vector = torch.cat(batch_meta_2_node_vector, dim=0) 
    return batch_node_idx,batch_meta_edge_idx,batch_edge_dist,batch_meta_2_node_idx,batch_meta_2_node_vector


class GatedMetaFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate1 = nn.Sequential(
            nn.Linear(2 * dim +3 , dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            # nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Linear(2 * dim +3 , dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            # nn.Sigmoid()
        )
        self.fuse = nn.Sequential(
            nn.Linear(dim , dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            # nn.Sigmoid()
        )

    def forward(self, res_feat, meta_feat, sec_ids,batch_pe_vector, batch_meta_2_node_edge, batch_meta_2_node_vector):
        meta_broadcast = meta_feat[sec_ids]  # (N, C)
        gate1_input = torch.cat([res_feat, meta_broadcast, -1* batch_pe_vector], dim=-1)
        gate1_weight = self.gate1(gate1_input)  # (N, 1)
        meta_add_broadcast = meta_feat[batch_meta_2_node_edge[0]]
        res_broadcast = res_feat[batch_meta_2_node_edge[1]]
        gated2_input = torch.cat([res_broadcast, meta_add_broadcast, batch_meta_2_node_vector], dim=-1)  # [E, 2C+3+1]
        gate2_weight = self.gate2(gated2_input)  # [E, 1]
        meta_add_fea = gate2_weight * meta_add_broadcast
        # import pdb;pdb.set_trace()
        meta_add_fea = scatter_mean(meta_add_fea, batch_meta_2_node_edge[1], dim=0)
        # import pdb;pdb.set_trace()
        fused = res_feat + gate1_weight * meta_broadcast + meta_add_fea
        return self.fuse(fused)
        # return self.fuse(torch.cat((fused,-1*batch_pe_vector),dim=-1))
    
class HierarchicalDeltaGNN(nn.Module):
    def __init__(self, in_dim, d_hidden_dim, latent_dim, sec_ids, meta_edge_index, edge_dist, pe_vector, meta_2_node_edge, meta_2_node_vector,out_dim):
        super().__init__()
        self.mapping_mlp = Linear(in_dim, out_dim //3 * latent_dim)
        # self.mapping_mlp = MLP(np.concatenate(([in_dim],d_hidden_dim)), np.concatenate((d_hidden_dim,[out_dim //3 * latent_dim])))
        self.mlp1 = Linear(latent_dim + 3, latent_dim)
        self.meta_gnn1 = GATv2Conv(latent_dim, latent_dim, edge_dim=1,heads=4, concat=False)
        self.fusion1 = GatedMetaFusion(latent_dim)
        self.mlp2 = Linear(latent_dim + 3, latent_dim)
        self.meta_gnn2 = GATv2Conv(latent_dim, latent_dim, edge_dim=1,heads=4, concat=False)
        self.fusion2 = GatedMetaFusion(latent_dim)
        # self.mlp3 = Linear(latent_dim, latent_dim)
        # self.meta_gnn3 = GATv2Conv(latent_dim, latent_dim, edge_dim=1,heads=4, concat=False)
        # self.fusion3 = GatedMetaFusion(latent_dim)
        self.sec_ids = sec_ids
        self.meta_edge_index = meta_edge_index
        self.edge_dist = edge_dist
        self.pe_vector = pe_vector
        self.meta_2_node_edge = meta_2_node_edge
        self.meta_2_node_vector = meta_2_node_vector
        self.point_num = out_dim // 3
        self.coord_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 3)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.mapping_mlp(x)
        batch_sec_idx,batch_meta_edge_idx,batch_edge_dist,batch_meta_2_node_idx,batch_meta_2_node_vector = expand_batch(B, self.sec_ids, self.meta_edge_index,self.edge_dist,self.meta_2_node_edge,self.meta_2_node_vector)
        batch_sec_idx, batch_meta_edge_idx, batch_edge_dist,batch_meta_2_node_idx,batch_meta_2_node_vector = batch_sec_idx.to(x.device), batch_meta_edge_idx.to(x.device), batch_edge_dist.to(x.device),batch_meta_2_node_idx.to(x.device),batch_meta_2_node_vector.to(x.device)
        # import pdb;pdb.set_trace()
        x = x.reshape(B * self.point_num, -1)
        batch_pe_vector = self.pe_vector.repeat(B,1).to(x.device)
        x = self.mlp1(torch.cat((x,batch_pe_vector),dim=-1))
        meta_x = scatter_mean(x, batch_sec_idx, dim=0)
        meta_x = self.meta_gnn1(meta_x, batch_meta_edge_idx, edge_attr= batch_edge_dist)
        x_fused = self.fusion1(x, meta_x, batch_sec_idx,batch_pe_vector,batch_meta_2_node_idx,batch_meta_2_node_vector)
        x = self.mlp2(torch.cat((x_fused,batch_pe_vector),dim=-1))
        meta_x = scatter_mean(x, batch_sec_idx, dim=0)
        meta_x = self.meta_gnn2(meta_x, batch_meta_edge_idx, edge_attr= batch_edge_dist)
        x_fused = self.fusion2(x, meta_x, batch_sec_idx,batch_pe_vector,batch_meta_2_node_idx,batch_meta_2_node_vector)
        # x = self.mlp3(x_fused)
        # meta_x = scatter_mean(x, batch_sec_idx, dim=0)
        # meta_x = self.meta_gnn3(meta_x, batch_meta_edge_idx, edge_attr= batch_edge_dist)
        # x_fused = self.fusion3(x, meta_x, batch_sec_idx)
        delta_pos = self.coord_head(x_fused).reshape(B,-1)
        return delta_pos
