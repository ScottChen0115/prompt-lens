"""Produce human-readable evidence from completed local training and held-out results."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
run=Path(json.loads((ROOT/'outputs/dataset-audit/active-run.json').read_text())['run_path'])
r=json.loads((run/'report.json').read_text());e=json.loads((run/'held-out-evaluation.json').read_text())
lines=['# Supplied-data bilingual model update','',f"Run: `{r['run_id']}`. XLM-RoBERTa, {r['parameters']:,} parameters (0.278B).",'',
'User explicitly authorized resuming local training and updating Prompt Lens. Both supplied datasets contain machine-generated labels. These results measure agreement with those labels, not independently verified user-input quality or downstream assistant success.','',
f"Trained on {r['dataset_counts']['train']:,} examples; selected epoch {r['selected_epoch']} of {r['epochs_completed']} completed epochs using 2,000 validation examples. Temperatures fitted on 2,000 separate calibration examples. Unknown labels were masked individually. One overlong training row was quarantined without changing original files.",'',
f"Training device: {r['device']}. Duration: {r['seconds']/60:.1f} minutes. Backbone and classifier changed; saved-model reload parity passed. Local promotion gates: {r['promotion_gate']}.",'',
'## Validation comparison','',
f"New model uncalibrated log loss: {r['uncalibrated_validation']['overall']['mean_log_loss']:.4f}; character TF-IDF baseline: {r['baseline_validation']['overall']['mean_log_loss']:.4f}. Lower is better. The new model is also compared with the previous fixture-trained model separately for both sources and languages.",'',
'## Frozen evaluation','',
'Model files, calibration temperatures and operating thresholds were frozen before the reserved test and challenge sets were opened. Evaluation runs on the same CPU float32 model as the application. Unsupported inputs are rejected instead of silently truncated.','',
'| Dataset | Supported / supplied | Macro F1 | Mean log loss | Clear-input false flags | Near development examples |',
'|---|---:|---:|---:|---:|---:|']
for name,item in e['results'].items():
 m=item['metrics']['overall'];rate=m['all_clear_false_flag_rate'];rate='n/a' if rate is None else f'{rate:.1%}'
 lines.append(f"| {name} | {item['supported_rows']} / {item['total_rows']} | {m['macro_f1']:.3f} | {m['mean_log_loss']:.4f} | {rate} | {item['near_development_similarity_at_least_0.9']} |")
lines+=['','Near-development means character TF-IDF cosine similarity ≥0.9, a diagnostic rather than proof of semantic equivalence. Detailed results include a slice below that similarity threshold, exact overlap counts, each language/source/label, and rejected record IDs. Shared templates and machine-label errors can inflate performance. There is no independent human-gold benchmark.','',
'## Application and limits','','Prompt Lens uses the saved model offline, with per-label temperature scaling and five experimental signals. The app does not generate a validated overall quality grade or imply a probability that GPT-6 Sol will misunderstand a request. Generic review questions are not model-generated explanations. Inputs remain limited to 12,000 characters and 256 tokenizer tokens, including context.','',
'Original datasets and the previous checkpoint are preserved. No supplied generator code was executed, no paid service was used, and no external deployment occurred.','',
'## Evidence','','- `report.json`: recipe, counts, validation/calibration, baselines, code/data hashes.','- `frozen-candidate.json`: pre-test model/calibration hash manifest.','- `held-out-evaluation.json`: final test/challenge metrics and limitations.','- `outputs/dataset-audit/audit.json`: source normalization and exact leakage audit.','- `outputs/dataset-audit/near-duplicate-diagnostic.json`: development similarity diagnostic.','']
(run/'Report.md').write_text('\n'.join(lines))
print(run/'Report.md')
