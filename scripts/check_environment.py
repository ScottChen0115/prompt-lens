"""Offline infrastructure check. Synthetic labels are not research evidence."""
import importlib.metadata
import json
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
torch.manual_seed(42)
torch.set_num_threads(4)
rows = [
    {"id": f"smoke-{i:02d}", "context": "one reminder", "request":
     "change reminder" if i % 2 else "keep reminder", "label": i % 2,
     "source": "synthetic_environment_fixture", "assistant_version": "none"}
    for i in range(20)
]
dataset = Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)
assert len(dataset) == 20
x = TfidfVectorizer().fit_transform(dataset["request"])
baseline = LogisticRegression(random_state=42).fit(x, dataset["label"])
assert np.isfinite(brier_score_loss(dataset["label"], baseline.predict_proba(x)[:, 1]))

report = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "purpose": "Environment smoke check only; no real data or pretrained weights",
    "python": platform.python_version(), "platform": platform.platform(),
    "architecture": platform.machine(), "fixture_rows": len(dataset),
    "packages": {p: importlib.metadata.version(p) for p in
                 ["torch", "transformers", "datasets", "accelerate", "numpy", "pandas",
                  "scikit-learn", "matplotlib", "safetensors", "pytest"]},
    "mps_built": torch.backends.mps.is_built(),
    "mps_available": torch.backends.mps.is_available(),
    "checks": {"dataset_ingestion": "passed", "tfidf_logistic_fit": "passed"},
    "devices": {},
}
with tempfile.TemporaryDirectory(prefix="risk-smoke-") as tmp:
    tmp = Path(tmp)
    (tmp / "vocab.txt").write_text("\n".join([
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "one", "reminder", "change", "keep"]) + "\n")
    tokenizer = BertTokenizerFast(vocab=str(tmp / "vocab.txt"), do_lower_case=True)
    # Verify tokenization really uses the fixture vocabulary.
    assert tokenizer.convert_tokens_to_ids("reminder") != tokenizer.unk_token_id
    batch = tokenizer(list(dataset["context"]), list(dataset["request"]),
                      padding="max_length", max_length=32, truncation=True, return_tensors="pt")
    labels = torch.tensor(dataset["label"], dtype=torch.float32)
    devices = ["cpu"] + (["mps"] if report["mps_available"] else [])
    for device in devices:
        config = BertConfig(vocab_size=len(tokenizer), hidden_size=128,
                            num_hidden_layers=2, num_attention_heads=2,
                            intermediate_size=512, max_position_embeddings=512,
                            num_labels=1)
        model = BertForSequenceClassification(config).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        inputs = {k: v.to(device) for k, v in batch.items()}
        before = model.classifier.weight.detach().cpu().clone()
        losses = []
        for _ in range(2):
            optimizer.zero_grad()
            logits = model(**inputs).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device))
            assert torch.isfinite(loss).item()
            loss.backward()
            assert all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            losses.append(loss.item())
        assert not torch.equal(before, model.classifier.weight.detach().cpu())
        model = model.cpu().eval()
        with torch.no_grad():
            expected = model(**batch).logits
        saved = tmp / device
        model.save_pretrained(saved, safe_serialization=True)
        tokenizer.save_pretrained(saved)
        reloaded = BertForSequenceClassification.from_pretrained(saved, local_files_only=True).eval()
        with torch.no_grad():
            actual = reloaded(**batch).logits
        torch.testing.assert_close(actual, expected)
        report["devices"][device] = {
            "forward_backward_optimizer": "passed", "save_reload_parity": "passed",
            "synthetic_losses_not_quality_metrics": losses,
        }

out = ROOT / "outputs/environment"
out.mkdir(parents=True, exist_ok=True)
plt.plot([0, 1], [0, 1])
plt.title("Plotting infrastructure check — not model calibration")
plt.savefig(out / "plot-smoke.png")
plt.close()
report["checks"]["plot_export"] = "passed"
(out / "environment-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
