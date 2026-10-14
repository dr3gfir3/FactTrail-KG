import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_model, get_pipeline
from src.wiki_llm_extractor import extract_with_llm


def main() -> int:
    parser = argparse.ArgumentParser(description="Run wiki LLM extraction with the real configured model.")
    parser.add_argument("--title", default="Marie Curie")
    parser.add_argument(
        "--text",
        default="Marie Curie discovered polonium and won the Nobel Prize in Physics.",
    )
    args = parser.parse_args()

    print("model:", get_model("llm"))
    print("max_new_tokens:", get_pipeline("llm_max_new_tokens"))
    print("temperature:", get_pipeline("llm_temperature"))
    print()

    try:
        result = extract_with_llm(args.text, args.title)
    except Exception as exc:
        print("ERROR: real model extraction failed")
        print(f"{type(exc).__name__}: {exc}")
        return 1

    triples = result["triples"]
    entities = result["entities"]
    print("entities:", entities)
    print("triples:")
    for triple in triples:
        print(" -", triple["triple"])
    print()
    print(result["wiki_content"])

    issues = []
    if not triples:
        issues.append("no triples parsed from model output")
    if not entities:
        issues.append("no entities parsed or inferred")

    text_lower = args.text.lower()
    for triple in triples:
        subject, _, obj = triple["triple"]
        if subject.lower() not in text_lower:
            issues.append(f"subject not grounded in source: {subject}")
        if obj.lower() not in text_lower:
            issues.append(f"object not grounded in source: {obj}")

    if issues:
        print("\nISSUES:")
        for issue in issues:
            print(" -", issue)
        return 2

    print("\nOK: model returned grounded entities and triples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
