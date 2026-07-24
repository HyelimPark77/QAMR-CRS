from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states.mean(dim=1)

    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden_states * mask).sum(dim=1) / denom


@dataclass
class ModalityEvidenceOutput:
    query: torch.Tensor
    weights: torch.Tensor
    evidence: Dict[str, torch.Tensor]
    shared: torch.Tensor


class QueryIntentEncoder(nn.Module):
    """A light query encoder built on top of existing text encoder outputs."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        token_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        entity_summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooled = masked_mean_pool(token_hidden_states, attention_mask)
        if entity_summary is None:
            entity_summary = torch.zeros_like(pooled)
        return self.proj(torch.cat([pooled, entity_summary], dim=-1))


class ModalityRouter(nn.Module):
    """Standard MLP gating used in adaptive multimodal fusion and MoE-style models."""

    def __init__(self, hidden_size: int, n_modalities: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_modalities),
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.mlp(query), dim=-1)


class EvidenceAggregator(nn.Module):
    """Project modality evidence into a shared space and aggregate with router weights."""

    def __init__(self, hidden_size: int, modality_names):
        super().__init__()
        self.modality_names = list(modality_names)
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            for name in self.modality_names
        })

    def forward(self, weights: torch.Tensor, evidence: Dict[str, torch.Tensor]) -> torch.Tensor:
        shared = None
        for idx, name in enumerate(self.modality_names):
            if name not in evidence:
                continue
            projected = self.projections[name](evidence[name])
            weight_shape = [weights.size(0)] + [1] * (projected.dim() - 1)
            weighted = projected * weights[:, idx].view(*weight_shape)
            shared = weighted if shared is None else shared + weighted

        if shared is None:
            raise ValueError("EvidenceAggregator received no matching evidence tensors.")
        return shared


class QueryAwareEvidenceRouter(nn.Module):
    """End-to-end wrapper used by rec/conv adapters."""

    def __init__(self, hidden_size: int, modality_names):
        super().__init__()
        self.modality_names = list(modality_names)
        self.query_encoder = QueryIntentEncoder(hidden_size)
        self.router = ModalityRouter(hidden_size, len(self.modality_names))
        self.aggregator = EvidenceAggregator(hidden_size, self.modality_names)

    def forward(
        self,
        token_hidden_states: torch.Tensor,
        modality_evidence: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        entity_summary: Optional[torch.Tensor] = None,
    ) -> ModalityEvidenceOutput:
        query = self.query_encoder(
            token_hidden_states=token_hidden_states,
            attention_mask=attention_mask,
            entity_summary=entity_summary,
        )
        weights = self.router(query)
        shared = self.aggregator(weights, modality_evidence)
        return ModalityEvidenceOutput(
            query=query,
            weights=weights,
            evidence=modality_evidence,
            shared=shared,
        )
