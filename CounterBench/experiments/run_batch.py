"""Run a single model, resuming from saved progress."""
import os, json, base64, time, sys
from pathlib import Path
import requests

model_name = sys.argv[1]
model_id = sys.argv[2]

API_KEY = os.environ.get("OPENROUTER_API_KEY")
DATA_DIR = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
RESULTS_DIR = Path(__file__).parent.parent / "results"

def encode_image(p):
    with open(p,"rb") as f: return base64.b64encode(f.read()).decode()

def query(mid, q, img_path):
    b64 = encode_image(img_path)
    h = {"Authorization":f"Bearer {API_KEY}","Content-Type":"application/json",
         "HTTP-Referer":"https://counterbench.research","X-Title":"CounterBench"}
    p = {"model":mid,"messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}},
        {"type":"text","text":q+"\n\nIMPORTANT: Give only a short, direct answer. Do not explain."}
    ]}],"max_tokens":50,"temperature":0.0}
    for a in range(3):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",headers=h,json=p,timeout=45)
            if r.status_code==429: time.sleep(5*(a+1)); continue
            if r.status_code>=400:
                if a<2: time.sleep(3); continue
                return f"ERROR:{r.status_code}"
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if a<2: time.sleep(3)
            else: return f"ERROR:{str(e)[:50]}"
    return "ERROR:retries"

with open(DATA_DIR/"benchmark.json") as f: bench = json.load(f)
rpath = RESULTS_DIR/f"{model_name}_results.json"

existing = {}
if rpath.exists():
    with open(rpath) as f:
        for r in json.load(f).get("results",[]):
            if not r.get("orig_response","").startswith("ERROR"):
                existing[r["id"]] = r

results, errs = [], 0
for i,item in enumerate(bench["items"]):
    if item["id"] in existing:
        results.append(existing[item["id"]]); continue
    
    o = query(model_id, item["question"], IMAGES_DIR/item["original_image"])
    time.sleep(0.2)
    iv = query(model_id, item["question"], IMAGES_DIR/item["intervened_image"])
    time.sleep(0.2)
    
    if o.startswith("ERROR") or iv.startswith("ERROR"): errs += 1
    results.append({"id":item["id"],"category":item["category"],"question":item["question"],
        "original_answer_gt":item["original_answer"],"intervened_answer_gt":item["intervened_answer"],
        "should_flip":item["should_flip"],"orig_response":o,"int_response":iv})
    
    if (i+1)%50==0:
        print(f"[{model_name}] {i+1}/550 ({errs} err, {len(existing)} cached)")
        with open(rpath,"w") as f: json.dump({"model":model_name,"model_id":model_id,"results":results},f)

with open(rpath,"w") as f: json.dump({"model":model_name,"model_id":model_id,"results":results},f)
print(f"[{model_name}] DONE: {len(results)} results, {errs} errors")
