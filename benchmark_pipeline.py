import sys, json, time
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

from datasets import load_dataset
from src.pipeline import verify

HOVER_DATASET = "Dzeniks/hover"

def bench(num_samples=10, split="train", seed=42, progress_callback=None):
    print(f"\n{'='*70}")
    print(f"PIPELINE BENCHMARK (Wiki-First GraphRAG)")
    print(f"Samples: {num_samples} | Split: {split}")
    print(f"{'='*70}\n")

    print("Loading HoVer dataset...")
    dataset = load_dataset(HOVER_DATASET, split=split)
    dataset = dataset.shuffle(seed=seed)
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    results = []
    correct = 0
    total = 0
    total_llm = 0
    correct_llm = 0
    total_faithful = 0
    faithful_count = 0
    confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    errors = []
    stage_totals = Counter()
    stage_counts = Counter()

    for i, row in enumerate(dataset):
        claim_text = row["claim"]
        gold_label = row["label"]
        claim_id = str(row["id"])

        print(f"\n[{i+1}/{num_samples}] {claim_text[:100]}...")
        print(f"  Gold: {'SUPPORTED' if gold_label == 0 else 'REFUTED'}")

        t_start = time.time()
        try:
            result = verify(claim_text, update_wiki=True)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            total += 1
            results.append({"id": claim_id, "claim": claim_text, "gold_label": gold_label,
                            "predicted": None, "correct": False, "error": str(e)})
            if progress_callback:
                try:
                    progress_callback(i + 1, num_samples, claim_text, "SUPPORTED" if gold_label == 0 else "REFUTED", "ERROR", False, {})
                except Exception:
                    pass
            continue

        pipeline_time = time.time() - t_start
        verdict = result["verdict"]
        predicted = 0 if verdict == "SUPPORTED" else 1
        is_correct = predicted == gold_label

        if is_correct:
            correct += 1
        total += 1

        if gold_label == 0 and predicted == 0:
            confusion["TP"] += 1
        elif gold_label == 0 and predicted == 1:
            confusion["FN"] += 1
        elif gold_label == 1 and predicted == 1:
            confusion["TN"] += 1
        else:
            confusion["FP"] += 1

        llm_verdict = result.get("llm_judge", {}).get("verdict")
        if llm_verdict is not None:
            total_llm += 1
            llm_predicted = 0 if llm_verdict == "SUPPORTED" else 1
            if llm_predicted == gold_label:
                correct_llm += 1

        # Faithfulness: check RoBERTa fallback was used
        roberta_out = result.get("roberta_fallback")
        if roberta_out:
            total_faithful += 1
            faithful_count += 1

        if not is_correct:
            err = {
                "id": claim_id, "claim": claim_text,
                "gold": "SUPPORTED" if gold_label == 0 else "REFUTED",
                "pred": verdict, "probs": result["probs"],
                "premise": result["premise"][:200],
            }
            if llm_verdict:
                err["llm_verdict"] = llm_verdict
                err["llm_reasoning"] = result.get("llm_judge", {}).get("reasoning", "")[:200]
            errors.append(err)

        for stage, val in result.get("stages", {}).items():
            if isinstance(val, (int, float)):
                stage_totals[stage] += val
                stage_counts[stage] += 1

        acc_sofar = correct / total * 100
        print(f"  Verdict: {verdict} {'OK' if is_correct else 'WRONG'} ({acc_sofar:.1f}%)")
        probs = result["probs"]
        print(f"  S={probs.get('SUPPORTED',0)*100:.0f}% R={probs.get('REFUTED',0)*100:.0f}% N={probs.get('NOT ENOUGH INFO',0)*100:.0f}%")
        print(f"  Pages added: {result['wiki_pages_added']} | KG triples: {result['wiki_triples']} | Time: {pipeline_time:.1f}s")

        results.append({
            "id": claim_id, "claim": claim_text, "gold_label": gold_label,
            "predicted": predicted, "verdict": verdict, "correct": is_correct,
            "probs": result["probs"], "pages_added": result["wiki_pages_added"],
            "triples": result["wiki_triples"], "time": pipeline_time,
            "llm_verdict": llm_verdict,
            "llm_reasoning": result.get("llm_judge", {}).get("reasoning", "")[:300],
            "roberta_fallback": roberta_out["verdict"] if roberta_out else None,
        })

        if progress_callback:
            try:
                progress_callback(i + 1, num_samples, claim_text, "SUPPORTED" if gold_label == 0 else "REFUTED", verdict, is_correct, result["probs"])
            except Exception:
                pass

    accuracy = correct / total * 100 if total else 0
    precision = confusion["TP"] / (confusion["TP"] + confusion["FP"]) * 100 if (confusion["TP"] + confusion["FP"]) else 0
    recall = confusion["TP"] / (confusion["TP"] + confusion["FN"]) * 100 if (confusion["TP"] + confusion["FN"]) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    llm_accuracy = correct_llm / total_llm * 100 if total_llm else 0

    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"Total:   {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall:   {recall:.2f}%")
    print(f"F1:       {f1:.2f}")
    print(f"Confusion: {confusion}")
    print(f"\nLLM-as-Judge: {correct_llm}/{total_llm} = {llm_accuracy:.1f}%")
    if total_faithful:
        print(f"RoBERTa fallback used: {faithful_count}/{total_faithful}")

    print(f"\nAvg Stage Times:")
    for stage in sorted(stage_totals.keys()):
        avg = stage_totals[stage] / max(stage_counts[stage], 1)
        print(f"  {stage:25s} {avg:.3f}s")

    if errors:
        print(f"\nErrors: {len(errors)}")
        ne = [e for e in errors if e["pred"] == "NOT ENOUGH INFO"]
        wrong = [e for e in errors if e["pred"] != "NOT ENOUGH INFO"]
        print(f"  NOT ENOUGH INFO: {len(ne)}")
        print(f"  Wrong verdict: {len(wrong)}")
        for e in errors[:3]:
            print(f"\n  [{e['id']}] Gold={e['gold']} Pred={e['pred']}")
            print(f"  Claim: {e['claim'][:100]}")
            print(f"  Premise: {e['premise'][:100]}")

    metrics = {
        "num_samples": total, "accuracy_pct": round(accuracy, 2),
        "precision_pct": round(precision, 2), "recall_pct": round(recall, 2),
        "f1_score": round(f1, 2), "correct": correct, "total": total,
        "confusion": confusion, "errors": len(errors),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_dir = Path("data/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pipeline_benchmark_{num_samples}samples.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results, "errors": errors},
                  f, ensure_ascii=False, indent=2)
    print(f"\nResults: {out_path}")
    return metrics, results


if __name__ == "__main__":
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    split = sys.argv[2] if len(sys.argv) > 2 else "train"
    bench(num_samples=num, split=split)
