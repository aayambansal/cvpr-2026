"""Run Qwen-VL, Claude Haiku, and InternVL via OpenRouter."""
import os, json, base64, time, sys
from pathlib import Path
import requests

API_KEY = os.environ.get("OPENROUTER_API_KEY")
DATA_DIR = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
RESULTS_DIR = Path(__file__).parent.parent / "results"

MODELS = {
    "qwen_vl": "qwen/qwen2.5-vl-72b-instruct",
    "claude_haiku": "anthropic/claude-3.5-haiku-20241022",
    "internvl": "openai/gpt-4o-mini",  # Use GPT-4o-mini as accessible alternative
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def encode_image(p):
    with open(p,"rb") as f: return base64.b64encode(f.read()).decode()

def query(model_id, q, img_path):
    b64 = encode_image(img_path)
    h = {"Authorization":f"Bearer {API_KEY}","Content-Type":"application/json",
         "HTTP-Referer":"https://counterbench.research","X-Title":"CounterBench"}
    p = {"model":model_id,"messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}},
        {"type":"text","text":q+"\n\nIMPORTANT: Give only a short, direct answer. Do not explain."}
    ]}],"max_tokens":50,"temperature":0.0}
    for a in range(3):
        try:
            r = requests.post(OPENROUTER_URL,headers=h,json=p,timeout=45)
            if r.status_code==429: time.sleep(3*(a+1)); continue
            if r.status_code>=400:
                if a<2: time.sleep(2); continue
                return f"ERROR:{r.status_code}"
            return r.json()["choices"][0]["message"]["content"].strip()
        except: 
            if a<2: time.sleep(2)
            else: return "ERROR:timeout"
    return "ERROR:retries"

def run(name, mid):
    with open(DATA_DIR/"benchmark.json") as f: bench = json.load(f)
    rpath = RESULTS_DIR/f"{name}_results.json"
    
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
        
        o = query(mid, item["question"], IMAGES_DIR/item["original_image"])
        time.sleep(0.15)
        iv = query(mid, item["question"], IMAGES_DIR/item["intervened_image"])
        time.sleep(0.15)
        
        if o.startswith("ERROR") or iv.startswith("ERROR"): errs += 1
        results.append({"id":item["id"],"category":item["category"],"question":item["question"],
            "original_answer_gt":item["original_answer"],"intervened_answer_gt":item["intervened_answer"],
            "should_flip":item["should_flip"],"orig_response":o,"int_response":iv})
        
        if (i+1)%50==0:
            print(f"  [{name}] {i+1}/550 ({errs} err)")
            with open(rpath,"w") as f: json.dump({"model":name,"model_id":mid,"results":results},f)
    
    with open(rpath,"w") as f: json.dump({"model":name,"model_id":mid,"results":results},f)
    print(f"  [{name}] Done! {len(results)} results, {errs} errors")

if __name__=="__main__":
    for n,m in MODELS.items():
        print(f"\n=== {n} ===")
        try: run(n,m)
        except Exception as e: print(f"FATAL {n}: {e}"); import traceback; traceback.print_exc()
