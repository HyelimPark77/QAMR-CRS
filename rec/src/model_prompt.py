import math
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv,GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from core import QueryAwareEvidenceRouter


class KGPrompt(nn.Module):
    def __init__(
        self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
        n_entity, num_relations, num_bases, edge_index, edge_type,
        n_prefix_rec=None, n_prefix_conv=None
    ):
        super(KGPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv

        entity_hidden_size = hidden_size // 2
        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations,
                                   num_bases=num_bases)
        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)

        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)

        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        entity_embeds = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def forward(self, entity_ids=None, token_embeds=None, output_entity=False, use_rec_prefix=False,
                use_conv_prefix=False):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds = self.get_entity_embeds()
            entity_embeds = entity_embeds[entity_ids]  # (batch_size, entity_len, hidden_size)
        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds  # (batch_size, token_len, hidden_size)
            token_embeds = self.token_proj2(token_embeds)

        if entity_embeds is not None and token_embeds is not None:
            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2,
                                                                                 1)  # (batch_size, token_len, entity_len)
            attn_weights /= self.hidden_size

            if output_entity:
                token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
                prompt_embeds = token_weights @ token_embeds + entity_embeds
                prompt_len = entity_len
            else:
                entity_weights = F.softmax(attn_weights, dim=2)
                prompt_embeds = entity_weights @ entity_embeds + token_embeds
                prompt_len = token_len
        elif entity_embeds is not None:
            prompt_embeds = entity_embeds
            prompt_len = entity_len
        else:
            prompt_embeds = token_embeds
            prompt_len = token_len

        if self.n_prefix_rec is not None and use_rec_prefix:
            prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_rec
        if self.n_prefix_conv is not None and use_conv_prefix:
            prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  # (n_layer, n_block, batch_size, n_head, prompt_len, head_dim)

        return prompt_embeds

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device('cpu')), strict=False
        )
        print(missing_keys, unexpected_keys)


class CustomGCNConv(MessagePassing):
    def __init__(self):
        super(CustomGCNConv, self).__init__(aggr='add')  # "Add" aggregation.

    def forward(self, x, edge_index):
        # 增加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 计算节点度数
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 执行消息传递
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # 消息传递（邻居信息聚合）
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # 直接返回聚合后的输出
        return aggr_out


class MMPrompt_inspired(nn.Module):
    def __init__(
        self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
        n_entity, num_relations, num_bases, edge_index, edge_type,edge_index_c,edge_index_t_s,edge_index_i_s, idx_to_id,
        n_prefix_rec=None, n_prefix_conv=None,
    ):
        super(MMPrompt_inspired, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv
        self.idx_to_id = idx_to_id
        self.idx_to_id_tensor = torch.tensor([self.idx_to_id[i] for i in range(len(self.idx_to_id))], dtype=torch.long)
        self.sorted_ids = sorted(self.idx_to_id.keys())
        self.sorted_indices = torch.tensor([self.idx_to_id[id] for id in self.sorted_ids], dtype=torch.long)
        entity_hidden_size = hidden_size // 2
        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations,num_bases=num_bases)
        self.conv_c1 = CustomGCNConv()  # LightGCN
        self.conv_c2 = CustomGCNConv()  # LightGCN
        self.conv_c3 = CustomGCNConv()  # LightGCN
        self.conv_ts1 = CustomGCNConv()  # LightGCN
        self.conv_ts2 = CustomGCNConv()  # LightGCN
        self.conv_ts3 = CustomGCNConv()  # LightGCN
        self.conv_is1 = CustomGCNConv()  # LightGCN
        self.conv_is2 = CustomGCNConv()  # LightGCN
        self.conv_is3 = CustomGCNConv()  # LightGCN

        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_index_c = nn.Parameter(edge_index_c,requires_grad=False)
        self.edge_index_t_s = nn.Parameter(edge_index_t_s,requires_grad=False)
        self.edge_index_i_s = nn.Parameter(edge_index_i_s,requires_grad=False)

        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)
        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)
        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)
        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        entity_embeds = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        sorted_indices = self.sorted_indices.to(entity_embeds.device)
        node_features = torch.index_select(entity_embeds, 0, sorted_indices)
        movie_embeds_ts1 = self.conv_ts1(node_features,self.edge_index_t_s)
        movie_embeds_ts2 = self.conv_ts2(movie_embeds_ts1,self.edge_index_t_s)
        movie_embeds_ts3 = self.conv_ts3(movie_embeds_ts2,self.edge_index_t_s)
        movie_embeds_mean_t = (movie_embeds_ts1 +movie_embeds_ts2)/2   

        movie_embeds_is1 = self.conv_is1(node_features,self.edge_index_i_s)
        movie_embeds_is2 = self.conv_is2(movie_embeds_is1,self.edge_index_i_s)
        movie_embeds_is3 = self.conv_is3(movie_embeds_is2,self.edge_index_i_s)
        movie_embeds_mean_i = (movie_embeds_is1+movie_embeds_is2)/2  
        movie_embeds_mean_t =(movie_embeds_mean_t + movie_embeds_mean_i)/2

        entity_embeds_c1 =self.conv_c1(entity_embeds, self.edge_index_c)
        entity_embeds_c2 =self.conv_c2(entity_embeds_c1, self.edge_index_c)
        entity_embeds_c3 =self.conv_c3(entity_embeds_c2, self.edge_index_c)

        #entity_embeds_c1 = self.conv_c1(node_embeds, self.edge_index_c)
        #entity_embeds_c2 = self.conv_c2(entity_embeds_c1, self.edge_index_c)
        #entity_embeds_c3 = self.conv_c3(entity_embeds_c2, self.edge_index_c)

        entity_embeds = (entity_embeds_c1 + entity_embeds_c2 + entity_embeds_c3+ entity_embeds) / 4
        device = movie_embeds_mean_t.device
        idx_to_id_tensor = self.idx_to_id_tensor.to(device)
        indices = idx_to_id_tensor[:len(movie_embeds_mean_t)]
        entity_embeds.index_add_(0, indices, movie_embeds_mean_t)
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def forward(self, entity_ids=None, token_embeds=None, output_entity=False, use_rec_prefix=False,
                use_conv_prefix=False):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds = self.get_entity_embeds()
            entity_embeds = entity_embeds[entity_ids] 
        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds  
            token_embeds = self.token_proj2(token_embeds)

        if entity_embeds is not None and token_embeds is not None:
            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2,
                                                                                 1)  
            attn_weights /= self.hidden_size

            if output_entity:
                token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
                prompt_embeds = token_weights @ token_embeds + entity_embeds
                token_weights_embeds = token_weights @ token_embeds  # 形状为 (batch_size, seq_len, num_entities)
                token_rep = token_weights_embeds.mean(dim=1)  # (batch_size, hidden_size)
                entity_rep = entity_embeds.mean(dim=1)  # (batch_size, hidden_size)
                temperature = 0.07
                logits = F.cosine_similarity(token_rep.unsqueeze(1), entity_rep.unsqueeze(0), dim=-1)
                logits /= temperature
                labels = torch.arange(logits.size(0), device=logits.device)  # (batch_size,)
                loss_cl = F.cross_entropy(logits, labels)
                prompt_len = entity_len
            else:
                entity_weights = F.softmax(attn_weights, dim=2)
                prompt_embeds = entity_weights @ entity_embeds + token_embeds
                prompt_len = token_len
        elif entity_embeds is not None:
            prompt_embeds = entity_embeds
            prompt_len = entity_len
        else:
            prompt_embeds = token_embeds
            prompt_len = token_len

        if self.n_prefix_rec is not None and use_rec_prefix:
            prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_rec
        if self.n_prefix_conv is not None and use_conv_prefix:
            prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  

        return prompt_embeds,loss_cl

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device('cpu')), strict=False
        )
        print(missing_keys, unexpected_keys)




class MMPrompt(nn.Module):
    def __init__(
        self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
        n_entity, num_relations, num_bases, edge_index, edge_type,edge_index_c,edge_index_t_s,edge_index_i_s, idx_to_id,
        n_prefix_rec=None, n_prefix_conv=None, routing_beta=0.0, pad_entity_id=None,
        learnable_routing_beta=False, entity_aware_routing=False,
    ):
        super(MMPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv
        self.routing_beta = routing_beta
        self.learnable_routing_beta = learnable_routing_beta
        self.entity_aware_routing = entity_aware_routing
        self.pad_entity_id = n_entity - 1 if pad_entity_id is None else pad_entity_id
        self.modality_names = ["kg", "co", "text", "image"]
        self.evidence_topk = 3
        if self.learnable_routing_beta:
            beta = min(max(float(routing_beta), 1e-4), 1.0 - 1e-4)
            self.routing_beta_logit = nn.Parameter(torch.logit(torch.tensor(beta)))

        self.idx_to_id = idx_to_id
        self.idx_to_id_tensor = torch.tensor([self.idx_to_id[i] for i in range(len(self.idx_to_id))], dtype=torch.long)
        self.sorted_ids = sorted(self.idx_to_id.keys())
        self.sorted_indices = torch.tensor([self.idx_to_id[id] for id in self.sorted_ids], dtype=torch.long)


        entity_hidden_size = hidden_size // 2
        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations,
                                   num_bases=num_bases)
        self.conv_c1 = CustomGCNConv()  # LightGCN
        self.conv_c2 = CustomGCNConv()  # LightGCN
        self.conv_c3 = CustomGCNConv()  # LightGCN
        self.conv_ts1 = CustomGCNConv()  # LightGCN
        self.conv_ts2 = CustomGCNConv()  # LightGCN
        self.conv_ts3 = CustomGCNConv()  # LightGCN
        self.conv_is1 = CustomGCNConv()  # LightGCN
        self.conv_is2 = CustomGCNConv()  # LightGCN
        self.conv_is3 = CustomGCNConv()  # LightGCN


        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_index_c = nn.Parameter(edge_index_c,requires_grad=False)
        self.edge_index_t_s = nn.Parameter(edge_index_t_s,requires_grad=False)
        self.edge_index_i_s = nn.Parameter(edge_index_i_s,requires_grad=False)

        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)

        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)
        self.evidence_router = QueryAwareEvidenceRouter(hidden_size, self.modality_names)
        # Keep initialization identical across routing ablations. The flag only
        # controls whether this router is used in forward().
        self.entity_router = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.last_router_weights = None
        self.last_selected_evidence = None
        self.last_router_entropy_loss = None
        self.last_router_balance_loss = None

        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def _project_entity_space(self, entity_tensor: torch.Tensor) -> torch.Tensor:
        entity_tensor = self.entity_proj1(entity_tensor) + entity_tensor
        return self.entity_proj2(entity_tensor)

    def get_modality_entity_embeds(self):
        node_embeds = self.node_embeds
        kg_entity_embeds = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        sorted_indices = self.sorted_indices.to(kg_entity_embeds.device)
        node_features = torch.index_select(kg_entity_embeds, 0, sorted_indices)
        movie_embeds_ts1 = self.conv_ts1(node_features,self.edge_index_t_s)
        movie_embeds_ts2 = self.conv_ts2(movie_embeds_ts1,self.edge_index_t_s)
        movie_embeds_ts3 = self.conv_ts3(movie_embeds_ts2,self.edge_index_t_s)
        movie_embeds_mean_t = (movie_embeds_ts1 + movie_embeds_ts2 + movie_embeds_ts3) / 3

        movie_embeds_is1 = self.conv_is1(node_features,self.edge_index_i_s)
        movie_embeds_is2 = self.conv_is2(movie_embeds_is1,self.edge_index_i_s)
        movie_embeds_is3 = self.conv_is3(movie_embeds_is2,self.edge_index_i_s)
        movie_embeds_mean_i = (movie_embeds_is1 + movie_embeds_is2 + movie_embeds_is3) / 3

        entity_embeds_c1 = self.conv_c1(node_embeds, self.edge_index_c)
        entity_embeds_c2 = self.conv_c2(entity_embeds_c1, self.edge_index_c)
        entity_embeds_c3 = self.conv_c3(entity_embeds_c2, self.edge_index_c)
        co_entity_embeds = (entity_embeds_c1 + entity_embeds_c2 + entity_embeds_c3) / 3

        device = movie_embeds_mean_t.device
        idx_to_id_tensor = self.idx_to_id_tensor.to(device)
        indices = idx_to_id_tensor[:len(movie_embeds_mean_t)]

        text_entity_embeds = torch.zeros_like(kg_entity_embeds)
        image_entity_embeds = torch.zeros_like(kg_entity_embeds)
        text_entity_embeds.index_add_(0, indices, movie_embeds_mean_t)
        image_entity_embeds.index_add_(0, indices, movie_embeds_mean_i)

        modality_embeds = {
            "kg": self._project_entity_space(kg_entity_embeds),
            "co": self._project_entity_space(co_entity_embeds),
            "text": self._project_entity_space(text_entity_embeds),
            "image": self._project_entity_space(image_entity_embeds),
        }
        fused_entity_embeds = sum(modality_embeds.values()) / len(modality_embeds)
        return modality_embeds, fused_entity_embeds

    def get_entity_embeds(self):
        _, fused_entity_embeds = self.get_modality_entity_embeds()
        return fused_entity_embeds

    def get_entity_embeds_for_scoring(self, router_weights=None):
        if router_weights is None:
            return self.get_entity_embeds()

        modality_entity_embeds, fused_entity_embeds = self.get_modality_entity_embeds()
        stacked = torch.stack([modality_entity_embeds[name] for name in self.modality_names], dim=0)
        weights = router_weights.to(device=stacked.device, dtype=stacked.dtype)
        routed_entity_embeds = torch.einsum("bm,mnh->bnh", weights, stacked)
        routing_beta = self.get_routing_beta()
        if torch.is_tensor(routing_beta) or routing_beta > 0:
            routed_entity_embeds = (
                routing_beta * fused_entity_embeds.unsqueeze(0)
                + (1.0 - routing_beta) * routed_entity_embeds
            )
        return routed_entity_embeds

    def get_last_router_weights(self):
        return self.last_router_weights

    def get_last_selected_evidence(self):
        return self.last_selected_evidence

    def get_last_router_entropy_loss(self):
        return self.last_router_entropy_loss

    def get_last_router_balance_loss(self):
        return self.last_router_balance_loss

    def get_routing_beta(self):
        if self.learnable_routing_beta:
            return torch.sigmoid(self.routing_beta_logit)
        return self.routing_beta

    def _entity_aware_route(self, query, modality_batch_embeds, entity_mask):
        projected = [
            self.evidence_router.aggregator.projections[name](modality_batch_embeds[name])
            for name in self.modality_names
        ]
        modality_stack = torch.stack(projected, dim=2)
        query_stack = query[:, None, None, :].expand(
            -1, modality_stack.size(1), modality_stack.size(2), -1
        )
        logits = self.entity_router(torch.cat([query_stack, modality_stack], dim=-1)).squeeze(-1)
        entity_router_weights = F.softmax(logits, dim=-1)
        routed_entity_embeds = (entity_router_weights.unsqueeze(-1) * modality_stack).sum(dim=2)

        mask = entity_mask.squeeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_weights = (entity_router_weights * mask.unsqueeze(-1)).sum(dim=1) / denom
        return routed_entity_embeds, mean_weights

    def _select_topk_evidence(self, query, modality_batch_embeds, entity_ids):
        selected = {}
        for name, embeds in modality_batch_embeds.items():
            scores = torch.matmul(embeds, query.unsqueeze(-1)).squeeze(-1)
            k = min(self.evidence_topk, scores.size(1))
            top_scores, top_indices = torch.topk(scores, k=k, dim=-1)
            top_entity_ids = torch.gather(entity_ids, 1, top_indices)
            selected[name] = {
                "entity_ids": top_entity_ids.detach().cpu(),
                "scores": top_scores.detach().cpu(),
            }
        return selected

    def forward(self, entity_ids=None, token_embeds=None, output_entity=False, use_rec_prefix=False,
                use_conv_prefix=False):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        modality_batch_embeds = None
        loss_cl = None
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            modality_entity_embeds, fused_entity_embeds = self.get_modality_entity_embeds()
            modality_batch_embeds = {
                name: embeds[entity_ids]
                for name, embeds in modality_entity_embeds.items()
            }
            entity_embeds = fused_entity_embeds[entity_ids]
        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds  
            token_embeds = self.token_proj2(token_embeds)
        if modality_batch_embeds is not None and token_embeds is not None:
            entity_mask = entity_ids.ne(self.pad_entity_id).unsqueeze(-1).to(entity_embeds.dtype)
            entity_summary = (entity_embeds * entity_mask).sum(dim=1) / entity_mask.sum(dim=1).clamp_min(1.0)
            router_output = self.evidence_router(
                token_hidden_states=token_embeds,
                modality_evidence=modality_batch_embeds,
                entity_summary=entity_summary,
            )
            if self.entity_aware_routing:
                routed_entity_embeds, router_weights = self._entity_aware_route(
                    router_output.query, modality_batch_embeds, entity_mask
                )
            else:
                routed_entity_embeds = router_output.shared
                router_weights = router_output.weights
            routing_beta = self.get_routing_beta()
            if torch.is_tensor(routing_beta) or routing_beta > 0:
                entity_embeds = routing_beta * entity_embeds + (1.0 - routing_beta) * routed_entity_embeds
            else:
                entity_embeds = routed_entity_embeds
            self.last_router_weights = router_weights.detach()
            self.last_router_entropy_loss = (
                router_weights * torch.log(router_weights.clamp_min(1e-12))
            ).sum(dim=-1).mean()
            mean_router = router_weights.mean(dim=0)
            self.last_router_balance_loss = (
                mean_router * torch.log(mean_router.clamp_min(1e-12))
            ).sum()
            self.last_selected_evidence = self._select_topk_evidence(
                query=router_output.query,
                modality_batch_embeds=modality_batch_embeds,
                entity_ids=entity_ids,
            )

        if entity_embeds is not None and token_embeds is not None:
            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2,
                                                                                 1)  
            attn_weights /= self.hidden_size

            if output_entity:
                token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
                prompt_embeds = token_weights @ token_embeds + entity_embeds

                token_weights_embeds = token_weights @ token_embeds  # 形状为 (batch_size, seq_len, num_entities)
                token_rep = token_weights_embeds.mean(dim=1)  # (batch_size, hidden_size)
                entity_rep = entity_embeds.mean(dim=1)  # (batch_size, hidden_size)
                temperature = 0.07
                logits = F.cosine_similarity(token_rep.unsqueeze(1), entity_rep.unsqueeze(0), dim=-1)
                logits /= temperature
                labels = torch.arange(logits.size(0), device=logits.device)  # (batch_size,)
                loss_cl = F.cross_entropy(logits, labels)
                prompt_len = entity_len
            else:
                entity_weights = F.softmax(attn_weights, dim=2)
                prompt_embeds = entity_weights @ entity_embeds + token_embeds
                prompt_len = token_len
        elif entity_embeds is not None:
            prompt_embeds = entity_embeds
            prompt_len = entity_len
        else:
            prompt_embeds = token_embeds
            prompt_len = token_len

        if self.n_prefix_rec is not None and use_rec_prefix:
            prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_rec
        if self.n_prefix_conv is not None and use_conv_prefix:
            prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  

        if loss_cl is None:
            loss_cl = prompt_embeds.new_zeros(())

        return prompt_embeds,loss_cl

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device('cpu')), strict=False
        )
        print(missing_keys, unexpected_keys)
