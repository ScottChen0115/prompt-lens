"""Activate a verified experimental local checkpoint; preserve the old pointer for rollback."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
run=Path(json.loads((ROOT/'outputs/dataset-audit/active-run.json').read_text())['run_path']).resolve()
assert run.is_relative_to((ROOT/'outputs/runs').resolve())
r=json.loads((run/'report.json').read_text());f=json.loads((run/'frozen-candidate.json').read_text())
e=json.loads((run/'held-out-evaluation.json').read_text())
assert r['status']=='completed_development_only' and all(r['promotion_gate'].values())
assert r['model_reload_parity']=='passed' and set(e['results'])=={'brq/test','rq/test','brq/challenge','rq/challenge'}
assert all(hashlib.sha256((run/'model'/name).read_bytes()).hexdigest()==value for name,value in f['model_hashes'].items())
assert e['frozen_manifest_sha256']==hashlib.sha256((run/'frozen-candidate.json').read_bytes()).hexdigest()
p=ROOT/'outputs/development/latest-bilingual.json'
backup=run/'previous-app-pointer.json'
if not backup.exists():backup.write_bytes(p.read_bytes())
new=p.with_suffix('.json.tmp');new.write_text(json.dumps({'run_path':str(run)},indent=2)+'\n');new.replace(p)
(run/'app-activation.json').write_text(json.dumps({'status':'pointer_updated','run_path':str(run),'experimental':True,'rollback_pointer':str(backup)},indent=2)+'\n')
print('Activated experimental checkpoint:',run.name)
