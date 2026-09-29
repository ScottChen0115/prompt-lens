"""Exercise the actual offline GUI worker with non-private bilingual examples."""
from pathlib import Path
import subprocess,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
run=Path(json.loads((ROOT/'outputs/development/latest-bilingual.json').read_text())['run_path'])
p=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/predict_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,cwd=ROOT)
ready=json.loads(p.stdout.readline());assert ready['type']=='ready' and ready['run_id']==run.name
records=[]
for payload in [{'request':'Write a three-bullet summary of the provided paragraph.','context':'The meeting is on Monday. The budget is approved. The launch is Friday.'},{'request':'请把下面的文字总结成三个要点。','context':'会议安排在周一。预算已经批准。产品将在周五发布。'},{'request':''},{'request':'Hello world. '*400}]:
 p.stdin.write(json.dumps(payload)+'\n');p.stdin.flush();result=json.loads(p.stdout.readline())
 records.append({'language':'zh' if payload['request'].startswith('请') else 'en','response':result})
assert all(r['response']['type']=='result' and len(r['response']['signals'])==5 for r in records[:2])
assert records[2]['response']['code']=='empty_request' and records[3]['response']['code']=='too_many_tokens'
latencies=[]
for i in range(20):
 payload={'request':'List three advantages and three disadvantages of remote work.','context':''}
 start=time.perf_counter();p.stdin.write(json.dumps(payload)+'\n');p.stdin.flush();result=json.loads(p.stdout.readline());latencies.append((time.perf_counter()-start)*1000)
 assert result['type']=='result'
p.stdin.close();p.wait(timeout=30);assert p.returncode==0
report={'ready':ready,'checks':records,'warm_cpu_latency_ms':{'samples':20,'median':float(np.median(latencies)),'p95':float(np.percentile(latencies,95))},'scope':'Short fixed request, warm worker, local CPU; not 256-token worst-case latency.'}
(run/'app-worker-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'run_id':ready['run_id'],'checks':'passed','latency':report['warm_cpu_latency_ms']}))
