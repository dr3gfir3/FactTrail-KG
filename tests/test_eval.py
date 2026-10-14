"""NLI accuracy tests — validates the core BERT-family linguistic model.

Runs inference on a shuffled sample from the HoVer test set and asserts
minimum accuracy thresholds. These tests require the model to be downloaded
once (~520MB for deberta-base-mnli).

Thresholds (above random 50%):
  - deberta-base-mnli (general MNLI): > 60% accuracy
  - deberta-base-long-nli (HoVer-tuned): > 70% accuracy
"""
import pytest
from src.eval import evaluate_nli, print_metrics


@pytest.mark.slow
@pytest.mark.parametrize("model,min_acc", [
    ("microsoft/deberta-base-mnli", 60.0),
    ("tasksource/deberta-base-long-nli", 70.0),
])
def test_nli_accuracy(model, min_acc):
    metrics, _ = evaluate_nli(model, num_samples=200)
    print_metrics(metrics)
    assert metrics["accuracy_pct"] >= min_acc, (
        f"{model}: {metrics['accuracy_pct']}% < {min_acc}% threshold"
    )
