# Caption NYU rgb512 images with Qwen3.8-27B-FP8 (vLLM, thinking disabled). Enforces SD3.5 CLIP token limit.
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import json, re, base64, io, os, glob, sys
DS = sys.argv[1]
import numpy as np
from vllm import LLM, SamplingParams
from transformers import CLIPTokenizer
SD = glob.glob(DATA + "/cache/huggingface/hub/models--stabilityai--stable-diffusion-3.5-medium/snapshots/*")[0]
toks = [CLIPTokenizer.from_pretrained(SD, subfolder=s) for s in ("tokenizer", "tokenizer_2")]
SUFFIX = ", architectural photography, straight lines, correct perspective"
def ntok(t): return max(len(tk(t, add_special_tokens=False).input_ids) for tk in toks)
SUF_N = ntok(SUFFIX); LIMIT = 75 - SUF_N
print("suffix tokens", SUF_N, "-> caption token limit", LIMIT, flush=True)
BANNED = re.compile(r"\b(blur\w*|nois\w*|grain\w*|resolution|pixel\w*|low[- ]quality|high[- ]quality|quality|sharp\w*|focus|perspective|vanishing|distort\w*|geometr\w*|straight lines?|warp\w*|artifact\w*|jpeg|compress\w*|exposure|overexposed|underexposed)\b", re.I)
PROMPT = ("Write one caption for this photo, to be used as a text-to-image prompt that recreates the same scene. "
          "Start with \"A photo of\". In at most 45 words, describe: the room type; the main objects and their materials and colors; "
          "the layout (what is where); the camera viewpoint (frontal view of a wall or looking into a corner, camera height such as eye level or low, "
          "and whether it tilts up or down); and the lighting. Do not mention image quality, sharpness, blur, noise, resolution, or the words "
          "perspective, geometry or straight lines. Output only the caption, as a single sentence or two.")
SHORTEN = "Rewrite this caption in at most {n} words, keeping room type, main objects, layout, viewpoint and lighting, starting with \"A photo of\" and never using the words {bad}. Output only the caption:\n\n{c}"
meta = json.load(open(DATA + f"/datasets/{DS}/meta.json"))["items"]
def durl(i):
    b = open(DATA + f"/datasets/{DS}/rgb512/{i:04d}.png", "rb").read()
    return "data:image/png;base64," + base64.b64encode(b).decode()
def main():
    llm = LLM(model="Qwen/Qwen3.8-27B-FP8", max_model_len=4096, limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=0.88, max_num_seqs=128, seed=0)
    CT = {"enable_thinking": False}
    def sp(seed): return SamplingParams(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=140, seed=seed)
    def clean(t):
        t = t.strip().strip('"').strip()
        t = re.sub(r"\s+", " ", t)
        return t
    def ok(t): return t.startswith("A photo of") and ntok(t) <= LIMIT and not BANNED.search(t)
    caps = {}; attempts = {}
    todo = [m["id"] for m in meta]
    for rnd in range(5):
        if not todo: break
        msgs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": durl(i)}}, {"type": "text", "text": PROMPT}]}] for i in todo]
        outs = llm.chat(msgs, [sp(1000 * rnd + i) for i in todo], chat_template_kwargs=CT, use_tqdm=True)
        nxt = []
        for i, o in zip(todo, outs):
            t = clean(o.outputs[0].text); attempts[i] = attempts.get(i, 0) + 1
            if ok(t): caps[i] = t
            else: nxt.append((i, t))
        print(f"round {rnd}: {len(todo)-len(nxt)} ok, {len(nxt)} need fixing", flush=True)
        # try shortening/rewording (text + image) before re-captioning from scratch
        if nxt:
            msgs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": durl(i)}},
                     {"type": "text", "text": SHORTEN.format(n=35, bad="quality, blur, noise, resolution, sharp, perspective, geometry, straight lines", c=t)}]}] for i, t in nxt]
            outs = llm.chat(msgs, [sp(77 + 1000 * rnd + i) for i, _ in nxt], chat_template_kwargs=CT, use_tqdm=False)
            todo = []
            for (i, _), o in zip(nxt, outs):
                t = clean(o.outputs[0].text); attempts[i] += 1
                if ok(t): caps[i] = t
                else: todo.append(i)
            print(f"  after shortening: {len(todo)} still failing", flush=True)
    dropped = list(todo)  # images that never produced a valid caption are excluded (recorded)
    print("DROPPED", dropped, flush=True)
    meta_ok = [m for m in meta if m['id'] in caps]
    n = np.array([ntok(caps[m["id"]]) for m in meta_ok]); ns = np.array([ntok(caps[m["id"]] + SUFFIX) for m in meta_ok])
    out = dict(model="Qwen/Qwen3.8-27B-FP8 (vLLM 0.33.0, thinking off, T=0.7 top_p=0.8 top_k=20 presence_penalty=1.5)", prompt=PROMPT,
               suffix=SUFFIX, token_limit_caption=LIMIT, captions={str(k): v for k, v in sorted(caps.items())}, dropped=dropped, attempts={str(k): v for k, v in attempts.items()})
    json.dump(out, open(DATA + f"/datasets/{DS}/captions.json", "w"), indent=1)
    print("TOKENS caption: median", np.median(n), "p95", np.percentile(n, 95), "max", n.max(), "| with suffix: median", np.median(ns), "max", ns.max())
    print("words: median", np.median([len(caps[m['id']].split()) for m in meta_ok]))
    print("CAPDONE")

if __name__ == '__main__':
    main()
