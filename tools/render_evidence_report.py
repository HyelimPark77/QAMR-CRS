"""Render human-readable reports from QAMR-CRS evidence logs."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPLANATION_PATH = PROJECT_ROOT / "core" / "explanation.py"
spec = importlib.util.spec_from_file_location("qamr_explanation", EXPLANATION_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

EvidenceGroundedExplainer = module.EvidenceGroundedExplainer
RedialEntityResolver = module.RedialEntityResolver


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Render human-readable evidence reports with names and explanations.")
    parser.add_argument("input", type=str, help="Path to evidence jsonl log.")
    parser.add_argument(
        "--mid2name",
        type=str,
        default=str(PROJECT_ROOT / "rec_data" / "redial" / "mid2name_redial.json"),
        help="Path to ReDial movie id-to-name map.",
    )
    parser.add_argument(
        "--id2entity",
        type=str,
        default=str(PROJECT_ROOT / "rec_data" / "redial" / "id2entity.jsonl"),
        help="Path to ReDial entity id-to-DBpedia map.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output markdown path. Defaults to <input>.report.md",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of records to render.",
    )
    args = parser.parse_args()

    resolver = RedialEntityResolver(args.mid2name, args.id2entity)
    explainer = EvidenceGroundedExplainer(resolver)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(input_path.suffix + ".report.md")

    records = load_records(input_path)[: args.limit]
    lines = ["# Evidence Report", ""]

    for idx, record in enumerate(records):
        lines.append(f"## Record {idx + 1}")
        lines.append("")
        lines.append(f"- split: {record.get('split')}")
        lines.append(f"- epoch: {record.get('epoch')}")
        lines.append(f"- step: {record.get('step')}")
        lines.append(f"- router_weights: {record.get('router_weights')}")
        labels = record.get("labels")
        if labels is not None:
            label_names = [resolver.resolve(int(label)) for label in labels]
            lines.append(f"- labels: {label_names}")
        lines.append(f"- explanation: {explainer.explain(record.get('router_weights'), record.get('selected_evidence'))}")
        lines.append("")
        lines.append("### Selected Evidence")
        lines.append("")
        selected = record.get("selected_evidence") or {}
        if not selected:
            lines.append("- none")
            lines.append("")
            continue
        for modality, payload in selected.items():
            lines.append(f"#### {modality}")
            entity_ids = payload.get("entity_ids", [])
            scores = payload.get("scores", [])
            flattened = []
            for batch_ids, batch_scores in zip(entity_ids, scores):
                for entity_id, score in zip(batch_ids, batch_scores):
                    flattened.append((int(entity_id), float(score)))
            if not flattened:
                lines.append("")
                lines.append("- none")
                lines.append("")
                continue
            lines.append("")
            for entity_id, score in flattened[:10]:
                lines.append(f"- {resolver.resolve(entity_id)} ({entity_id}), score={score:.4f}")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
