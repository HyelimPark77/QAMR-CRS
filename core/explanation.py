import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _clean_dbpedia_uri(uri: Optional[str]) -> Optional[str]:
    if uri is None:
        return None
    name = uri.strip()
    if name.startswith("<http://dbpedia.org/resource/") and name.endswith(">"):
        name = name[len("<http://dbpedia.org/resource/") : -1]
    name = name.replace("_", " ")
    name = re.sub(r"\s*\([^)]*\)", "", name)
    name = re.sub(r"__\d+$", "", name)
    return name.strip() or None


class RedialEntityResolver:
    """Resolve ReDial item/entity ids into readable names."""

    def __init__(
        self,
        mid2name_path: str,
        id2entity_path: str,
    ):
        self.item_names = json.loads(Path(mid2name_path).read_text(encoding="utf-8"))
        self.id2entity = json.loads(Path(id2entity_path).read_text(encoding="utf-8"))

    def resolve(self, entity_id: int) -> str:
        key = str(entity_id)
        if key in self.item_names:
            return self.item_names[key]
        if key in self.id2entity:
            cleaned = _clean_dbpedia_uri(self.id2entity[key])
            if cleaned:
                return cleaned
        return f"entity_{entity_id}"


class EvidenceGroundedExplainer:
    """Template-based grounded explanation generator for early-stage analysis."""

    modality_templates = {
        "kg": "지식그래프 근거로는 {names}가 현재 질의와 가장 강하게 연결되었습니다.",
        "co": "공기출현 근거로는 {names}가 대화 맥락과 함께 자주 등장하는 후보로 선택되었습니다.",
        "text": "텍스트 의미 근거로는 {names}가 현재 질의와 의미적으로 가장 가까운 증거였습니다.",
        "image": "시각 근거로는 {names}가 현재 질의와 가까운 분위기 또는 시각적 유사성을 보였습니다.",
    }

    def __init__(self, resolver: RedialEntityResolver):
        self.resolver = resolver

    def _top_modality(self, router_weights: Optional[Iterable[float]]) -> Optional[str]:
        if router_weights is None:
            return None
        weights = list(router_weights)
        if not weights:
            return None
        modality_names = ["kg", "co", "text", "image"]
        max_idx = max(range(len(weights)), key=lambda idx: weights[idx])
        return modality_names[max_idx]

    def _resolve_names(self, selected_evidence: Dict, modality: str, topn: int = 3) -> List[str]:
        payload = selected_evidence.get(modality) if selected_evidence else None
        if not payload:
            return []
        ids = payload.get("entity_ids", [])
        flat_ids = []
        for batch_ids in ids:
            for entity_id in batch_ids:
                flat_ids.append(entity_id)
                if len(flat_ids) >= topn:
                    break
            if len(flat_ids) >= topn:
                break
        return [self.resolver.resolve(int(entity_id)) for entity_id in flat_ids]

    def explain(self, router_weights: Optional[Iterable[float]], selected_evidence: Optional[Dict]) -> str:
        modality = self._top_modality(router_weights)
        if modality is None or not selected_evidence:
            return "선택된 evidence가 없어 설명을 생성할 수 없습니다."

        names = self._resolve_names(selected_evidence, modality)
        if not names:
            return "선택된 evidence가 없어 설명을 생성할 수 없습니다."

        joined = ", ".join(names)
        return self.modality_templates[modality].format(names=joined)
