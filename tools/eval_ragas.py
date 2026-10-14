"""
RAGAS-based evaluation of the GraphRAG fact-checking pipeline.

Usage:
  python tools/eval_ragas.py [num_claims] [--ragas-only]

With --ragas-only, loads cached pipeline results and runs RAGAS eval.
"""
import json, sys, os, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HOVER_DATASET = "Dzeniks/hover"
EVAL_DIR = Path("data/eval")
EVAL_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = EVAL_DIR / "pipeline_cache_30.json"
LABEL_MAP = {0: "SUPPORTED", 1: "REFUTED"}

def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def extract_top_evidence(result, top_k=5):
    texts = []
    trace = result.get("evidence_trace", [])
    if trace:
        for ev in trace[:top_k]:
            if isinstance(ev, dict) and "text" in ev and ev["text"]:
                texts.append(ev["text"])
            elif isinstance(ev, dict) and "linearized" in ev:
                texts.append(ev["linearized"])
            elif isinstance(ev, str):
                texts.append(ev)
    if not texts:
        premise = result.get("premise", "")
        if premise:
            texts = [premise[:500]]
    if not texts:
        nli_ev = result.get("nli_decision_evidence", "")
        if nli_ev:
            texts = [nli_ev[:500]]
    return texts

def run_pipeline_batch(num_samples):
    from datasets import load_dataset
    import src.pipeline as P

    log(f"Loading HoVer test split, selecting {num_samples} samples...")
    dataset = load_dataset(HOVER_DATASET, split="test")
    dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    log(f"Loaded {len(dataset)} claims")

    questions, answers, contexts, ground_truths = [], [], [], []
    results_raw = []
    nli_correct = 0
    total_with_evidence = 0

    t_start = time.time()
    seen_ids = set()

    for i, row in enumerate(dataset):
        claim = row["claim"]
        gold_label = LABEL_MAP[row["label"]]
        claim_id = str(row["id"])
        if claim_id in seen_ids:
            continue
        seen_ids.add(claim_id)
        t_claim = time.time()

        log(f"\n[{i+1}/{num_samples}] claim_id={claim_id} gold={gold_label}")
        log(f"  claim: {claim[:120]}...")

        try:
            result = P.verify(claim, update_wiki=False)
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()
            results_raw.append({"claim_id": claim_id, "claim": claim, "error": str(e)})
            # Save partial progress
            _save_partial(questions, answers, contexts, ground_truths, results_raw,
                         nli_correct, total_with_evidence, t_start)
            continue

        elapsed = time.time() - t_claim
        llm_judge = result.get("llm_judge", {})
        llm_verdict = llm_judge.get("verdict")
        verdict = result.get("verdict", "UNKNOWN")
        nli_verdict = result.get("verdict", "")
        reasoning = llm_judge.get("reasoning", "")
        answer_text = f"VERDICT: {verdict}\nREASONING: {reasoning}"
        top_ctx = extract_top_evidence(result, top_k=5)

        has_ev = len(top_ctx) > 0
        if has_ev:
            total_with_evidence += 1
            if nli_verdict == gold_label:
                nli_correct += 1

        log(f"  verdict={verdict} llm={llm_verdict} gold={gold_label} evidence={has_ev} ({elapsed:.1f}s)")

        questions.append(claim)
        answers.append(answer_text)
        contexts.append(top_ctx)
        ground_truths.append(gold_label)

        results_raw.append({
            "claim_id": claim_id,
            "claim": claim,
            "gold_label": gold_label,
            "verdict": nli_verdict,
            "confidence": result.get("confidence"),
            "llm_verdict": llm_verdict,
            "llm_reasoning": reasoning,
            "evidence": top_ctx,
            "has_evidence": has_ev,
            "stages": result.get("stages", {}),
        })

        # Save after each claim
        _save_partial(questions, answers, contexts, ground_truths, results_raw,
                     nli_correct, total_with_evidence, t_start)

    total_elapsed = time.time() - t_start
    log(f"\n=== Pipeline complete: {len(questions)} claims in {total_elapsed:.1f}s ===")
    log(f"Average: {total_elapsed/max(len(questions),1):.1f}s per claim")
    log(f"Evidence found: {total_with_evidence}/{len(questions)}")
    log(f"NLI accuracy (on non-empty): {nli_correct}/{total_with_evidence} = {100*nli_correct/max(total_with_evidence,1):.1f}%")

    return questions, answers, contexts, ground_truths, results_raw

def _save_partial(questions, answers, contexts, ground_truths, results_raw,
                  nli_correct=0, total_with_evidence=0, t_start=0):
    data = {
        "num_processed": len(questions),
        "questions": questions,
        "answers": answers,
        "contexts": contexts,
        "ground_truths": ground_truths,
        "results_raw": results_raw,
        "interim": {
            "nli_correct": nli_correct,
            "total_with_evidence": total_with_evidence,
        }
    }
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def run_ragas_eval(questions, answers, contexts, ground_truths):
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import answer_correctness
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.run_config import RunConfig
    from langchain_ollama import ChatOllama

    # Filter to non-empty contexts
    valid = [(q, a, ctx, ref) for q, a, ctx, ref
             in zip(questions, answers, contexts, ground_truths) if ctx]
    if not valid:
        log("No claims with non-empty contexts for RAGAS evaluation.")
        return {}, None

    log(f"\nFiltered to {len(valid)} claims with non-empty evidence for RAGAS eval.")

    log("Setting up RAGAS evaluator (Mistral via Ollama)...")
    llm = LangchainLLMWrapper(
        ChatOllama(model="ministral-3:3b", temperature=0.0, num_predict=512, num_ctx=4096, timeout=600)
    )
    emb = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2", device="cpu")

    samples_list = []
    for q, a, ctx, ref in valid:
        samples_list.append({
            "user_input": q,
            "response": a,
            "retrieved_contexts": ctx,
            "reference": ref,
        })

    dataset = EvaluationDataset.from_dict(samples_list)
    log(f"Evaluating {len(samples_list)} samples (single metric: answer_correctness)...")

    t_ragas = time.time()
    run_config = RunConfig(timeout=600, max_retries=3)
    result = evaluate(
        dataset,
        metrics=[answer_correctness],
        llm=llm,
        embeddings=emb,
        run_config=run_config,
        raise_exceptions=False,
        batch_size=1,
    )
    ragas_elapsed = time.time() - t_ragas
    log(f"RAGAS evaluation completed in {ragas_elapsed:.1f}s")

    log("\n" + "="*60)
    log("RAGAS METRICS")
    log("="*60)
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        log(df.to_string())
        scores = {}
        for col in df.columns:
            if col in ("answer_correctness",):
                scores[col] = float(df[col].mean())
        log("\nAggregate:")
        for k, v in scores.items():
            log(f"  {k}: {v:.4f}")
        return scores, df
    return {}, None

def main():
    args = sys.argv[1:]
    ragas_only = "--ragas-only" in args
    num_samples = 30

    for a in args:
        if a.isdigit():
            num_samples = int(a)
            break

    if ragas_only and CACHE_PATH.exists():
        log(f"Loading cached pipeline results from {CACHE_PATH}...")
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        questions = data.get("questions", [])
        answers = data.get("answers", [])
        contexts = data.get("contexts", [])
        ground_truths = data.get("ground_truths", [])
        results_raw = data.get("results_raw", [])
        log(f"Loaded {len(questions)} claims from cache.")
    else:
        questions, answers, contexts, ground_truths, results_raw = run_pipeline_batch(num_samples)

    scores = {}
    if len(questions) >= 2:
        scores, df = run_ragas_eval(questions, answers, contexts, ground_truths)

    # Build final output
    nli_correct = sum(1 for r in results_raw if r.get("has_evidence") and r.get("verdict") == r.get("gold_label"))
    total_ev = sum(1 for r in results_raw if r.get("has_evidence"))
    output = {
        "config": {"num_samples": num_samples},
        "summary": {
            "total_claims": len(results_raw),
            "with_evidence": total_ev,
            "nli_accuracy": round(nli_correct / max(total_ev, 1), 4),
            "llm_judge_verdicts": {},
        },
        "ragas_scores": scores,
        "raw_results": results_raw,
    }

    # Count LLM judge verdicts
    for r in results_raw:
        lv = r.get("llm_verdict", "UNKNOWN")
        output["summary"]["llm_judge_verdicts"][lv] = output["summary"]["llm_judge_verdicts"].get(lv, 0) + 1

    out_path = EVAL_DIR / "ragas_eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
