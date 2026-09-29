"""Private stdin/stdout inference worker. Does not train, persist prompts or use the network."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# Keep all model loading offline; this worker never downloads or updates weights.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HOME'] = str(ROOT/'.cache/huggingface')
sys.path.insert(0,str(ROOT/'src'))
from risk_model.inference import QualityPredictor, InputError


def send(value):
    print(json.dumps(value,ensure_ascii=False),flush=True)


def main():
    try:
        predictor = QualityPredictor()
    except Exception:
        send({'type':'fatal','code':'model_load_failed'})
        return 1
    send(predictor.metadata)
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            result = predictor.predict(payload)
            result['id'] = payload.get('id')
            send(result)
        except InputError as error:
            send({'type':'error','code':error.code,**error.details})
        except (json.JSONDecodeError, TypeError):
            send({'type':'error','code':'invalid_input'})
        except Exception:
            send({'type':'error','code':'inference_failed'})
    return 0

if __name__=='__main__':
    raise SystemExit(main())
