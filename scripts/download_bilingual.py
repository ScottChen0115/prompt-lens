"""Fetch the pinned public base into the local cache. No account token is used."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT=Path(__file__).resolve().parents[1]
repo='FacebookAI/xlm-roberta-base'
revision='e73636d4f797dec63c3081bb6ed5c7b0bb3f2089'
path=snapshot_download(repo,revision=revision,token=False,
    allow_patterns=['config.json','tokenizer.json','tokenizer_config.json',
                    'sentencepiece.bpe.model','model.safetensors','README.md'])
config={'repo':repo,'revision':revision,'local_path':path,'license':'mit',
        'purpose':'bilingual_request_quality','parameter_range':[100000000,300000000]}
(ROOT/'configs/checkpoint-bilingual.json').write_text(json.dumps(config,indent=2)+'\n')
print(path)
