import json
import os
from collections import defaultdict

import numpy as np
import torch
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity


class DBpedia:
    def __init__(self, dataset, debug=False):
        self.debug = debug
        self.dataset_dir = os.path.join("rec_data", dataset)
        with open(os.path.join(self.dataset_dir, "dbpedia_subkg.json"), "r", encoding="utf-8") as f:
            self.entity_kg = json.load(f)
        with open(os.path.join(self.dataset_dir, "entity2id.json"), "r", encoding="utf-8") as f:
            self.entity2id = json.load(f)
        with open(os.path.join(self.dataset_dir, "relation2id.json"), "r", encoding="utf-8") as f:
            self.relation2id = json.load(f)
        with open(os.path.join(self.dataset_dir, "item_ids.json"), "r", encoding="utf-8") as f:
            self.item_ids = json.load(f)
        self._process_entity_kg()

    def _process_entity_kg(self, self_loop_id=185):
        n_entity = len(self.entity2id)
        edge_list = []

        for entity in range(n_entity + 1):
            edge_list.append((entity, entity, self_loop_id))
            if str(entity) not in self.entity_kg:
                continue
            for relation, tail in self.entity_kg[str(entity)]:
                if entity != tail and relation != self_loop_id:
                    edge_list.append((entity, tail, relation))
                    edge_list.append((tail, entity, relation))

        relation_cnt = defaultdict(int)
        relation_idx = {}
        for _, _, relation in edge_list:
            relation_cnt[relation] += 1
        for _, _, relation in edge_list:
            if relation_cnt[relation] > 1000 and relation not in relation_idx:
                relation_idx[relation] = len(relation_idx)

        filtered_edges = [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 1000]
        edge = torch.as_tensor(filtered_edges, dtype=torch.long)
        self.edge_index = edge[:, :2].t()
        self.edge_type = edge[:, 2]
        self.num_relations = len(relation_idx)
        self.pad_entity_id = max(self.entity2id.values()) + 1
        self.num_entities = max(self.entity2id.values()) + 2

        if self.debug:
            logger.debug(
                f"#edge: {len(edge)}, #relation: {self.num_relations}, "
                f"#entity: {self.num_entities}, #item: {len(self.item_ids)}"
            )

    def get_entity_kg_info(self):
        return {
            "edge_index": self.edge_index,
            "edge_type": self.edge_type,
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "pad_entity_id": self.pad_entity_id,
            "item_ids": self.item_ids,
        }


class Co_occurrence:
    def __init__(self, dataset, split, entity_max_length, all_items, n_entity, debug=False):
        self.debug = debug
        input_file = os.path.join("rec_data", dataset, "edge_index_c.pt")
        self.edge_index_c = torch.load(input_file)

    def get_entity_co_info(self):
        return {"edge_index_c": self.edge_index_c}


class text_sim:
    def __init__(self, dataset, pad_entity_id):
        dataset_dir = os.path.join("rec_data", dataset)
        data_file = os.path.join(dataset_dir, "id_embeddings_text.json")
        self.pad_entity_id = pad_entity_id
        self.prepare_data(data_file)

    def prepare_data(self, data_file):
        with open(data_file, "r", encoding="utf-8") as f:
            id_embeddings = json.load(f)
        new_key = self.pad_entity_id
        id_embeddings[str(new_key)] = [1.0] * 768
        self.keys = [int(key) for key in id_embeddings.keys()]
        self.id_to_idx = {node_id: idx for idx, node_id in enumerate(self.keys)}
        self.idx_to_id = {idx: node_id for node_id, idx in self.id_to_idx.items()}
        embeddings = np.array(list(id_embeddings.values()))
        similarity_matrix = cosine_similarity(embeddings)
        top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, 1:21]
        mapped_edges = []
        for i, key in enumerate(self.keys):
            src_idx = self.id_to_idx[key]
            for idx in top_k_indices[i]:
                mapped_edges.append([src_idx, self.id_to_idx[self.keys[idx]]])
        self.edge_index_t_s = torch.as_tensor(
            [[edge[0] for edge in mapped_edges], [edge[1] for edge in mapped_edges]],
            dtype=torch.long,
        )

    def get_entity_ts_info(self):
        return {
            "edge_index_t_s": self.edge_index_t_s,
            "id_to_idx": self.id_to_idx,
            "idx_to_id": self.idx_to_id,
            "all_movie": self.keys,
        }


class image_sim:
    def __init__(self, dataset, pad_entity_id):
        dataset_dir = os.path.join("rec_data", dataset)
        data_file = os.path.join(dataset_dir, "id_embeddings_image.json")
        self.pad_entity_id = pad_entity_id
        self.prepare_data(data_file)

    def prepare_data(self, data_file):
        with open(data_file, "r", encoding="utf-8") as f:
            id_embeddings = json.load(f)
        new_key = self.pad_entity_id
        id_embeddings[str(new_key)] = [1.0] * 768
        self.keys = [int(key) for key in id_embeddings.keys()]
        self.id_to_idx = {node_id: idx for idx, node_id in enumerate(self.keys)}
        self.idx_to_id = {idx: node_id for node_id, idx in self.id_to_idx.items()}
        embeddings = np.array(list(id_embeddings.values()))
        similarity_matrix = cosine_similarity(embeddings)
        top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, 1:21]
        mapped_edges = []
        for i, key in enumerate(self.keys):
            src_idx = self.id_to_idx[key]
            for idx in top_k_indices[i]:
                mapped_edges.append([src_idx, self.id_to_idx[self.keys[idx]]])
        self.edge_index_i_s = torch.as_tensor(
            [[edge[0] for edge in mapped_edges], [edge[1] for edge in mapped_edges]],
            dtype=torch.long,
        )

    def get_entity_is_info(self):
        return {
            "edge_index_i_s": self.edge_index_i_s,
            "id_to_idx": self.id_to_idx,
            "idx_to_id": self.idx_to_id,
            "all_movie": self.keys,
        }
