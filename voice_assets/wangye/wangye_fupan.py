# -*- coding: utf-8 -*-
"""王也念A股复盘：同《狼》配方（wy_long48锚/cfg2.5/t40/分块校验/atempo0.92+扩展器）"""
import os, sys, time, json, subprocess
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "/home/wooluo/voxemw-deploy/voxemw-app")
import numpy as np, soundfile as sf, librosa

T0 = time.perf_counter()
from voxcpm import VoxCPM
m = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=True)
SR = m.tts_model.sample_rate
print(f"[load] {time.perf_counter()-T0:.1f}s", flush=True)

ANCHOR = "/tmp/wy_long48.wav"

def measure(y, sr):
    f0,_,_ = librosa.pyin(y=y, fmin=50, fmax=500, sr=sr, frame_length=2048)
    v = f0[~np.isnan(f0)]
    r = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = 20*np.log10(r+1e-9); keep = db > db.max()-60
    d = np.sort(db[keep]); n = max(len(d),2)
    return ((float(v.mean()) if v.size else 0.), (float(v.std()) if v.size else 0.),
            float(d[int(.97*n)]-d[int(.03*n)]), float(len(y)/sr))

a,_ = sf.read(ANCHOR); a = a.astype(np.float32)
if a.ndim > 1: a = a.mean(axis=1)
BM, BS, BD, _ = measure(a, SR)
print(f"[锚基线] F0={BM:.0f} ±{BS:.0f} dyn={BD:.0f}dB", flush=True)

CHUNKS = [
 ("c1","九月三十，节前最后一个交易日，收盘了。过来坐，我给你们说道说道。"),
 ("c2","上证三千八百四十二，红了零点三个点。可你细看，科创五十跌了两点五，半导体直接躺了三个点。"),
 ("c3","银行倒涨了一点四，地产也泛红。嘿，钱从高处撤下来，躲进低估值的窝里过节去了。都学精了。"),
 ("c4","两市一万四千多亿，该走的今天都走干净了。留下来的，是真心想拿着过节的。"),
 ("c5","有人问我怎么看后市。我说啊，这就是个避险，不是变心。科技那点儿获利，兑现一下，不丢人。"),
 ("c6","节后看两样，量回不回来，科技敢不敢再领涨。回不来，就接着喝茶看戏。"),
 ("c7","拿着的，拿稳了。空仓的，正好歇歇。追高银行这种事儿，就别干了，何必呢。"),
 ("c8","行了，就说到这儿。都散了吧，过节去。风里雨里，节后再见。"),
]

cache = m.tts_model.build_prompt_cache(reference_wav_path=ANCHOR)

def gen(text):
    buf = []
    for item in m.tts_model.generate_with_prompt_cache_streaming(
            target_text=text, prompt_cache=cache,
            cfg_value=2.5, inference_timesteps=40):
        w = item[0] if isinstance(item, tuple) else item
        buf.append(np.asarray(w, dtype=np.float32).reshape(-1))
    return np.concatenate(buf)

res, picked = [], []
for tag, text in CHUNKS:
    nc = len(text); best = None
    for k in range(1, 4):
        y = gen(text)
        name = f"/tmp/wyf_{tag}_t{k}.wav"
        sf.write(name, y, SR)
        fm, fs, dyn, dur = measure(y, SR)
        dur_ok = (nc/9.0) <= dur <= (nc/1.8)
        ok = (BM-30) <= fm <= (BM+35) and fs >= 15 and dur_ok
        score = abs(fm-BM)/BM + abs(fs-BS)/max(BS,20) + (0 if dur_ok else 1.0)
        res.append(dict(file=name, tag=tag, k=k, f0=round(fm,1), std=round(fs,1),
                        dyn=round(dyn,1), dur=round(dur,2), ok=ok, score=round(score,3)))
        print(f"{name}  F0={fm:.0f} ±{fs:.0f}  dyn={dyn:.0f}dB  {dur:.1f}s({nc}字)  "
              f"{'OK' if ok else 'REJ'}", flush=True)
        if ok and (best is None or score < best[0]): best = (score, y)
        if ok: break
    if best is None:
        print(f"[FAIL] {tag} 三抽皆崩，跳段", flush=True); continue
    picked.append(best[1])

json.dump(res, open("/tmp/wyf_metrics.json","w"), ensure_ascii=False, indent=1)
if not picked:
    print("NO VALID CHUNKS", flush=True); sys.exit(1)

out = []
for i, y in enumerate(picked):
    if i: out.append(np.zeros(int(0.75*SR), dtype=np.float32))
    out.append(y)
raw = np.concatenate(out)
sf.write("/tmp/wyf_raw.wav", raw, SR)
print(f"[拼接] {len(picked)}/{len(CHUNKS)}段  raw={len(raw)/SR:.1f}s", flush=True)

subprocess.run(["ffmpeg","-y","-i","/tmp/wyf_raw.wav","-filter:a","atempo=0.92",
                "/tmp/wyf_slow.wav"], check=True, capture_output=True)
y,_ = sf.read("/tmp/wyf_slow.wav"); y = y.astype(np.float32)

def expand(y, sr, thr=-24.0, ratio=2.5, floor_db=-65.0):
    r = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = 20*np.log10(r+1e-9)
    g = np.where(db > thr, 0.0, np.maximum((db-thr)*(1-1/ratio), floor_db-db))
    hop = 512; g_s = np.copy(g)
    a_win, r_win = max(int(0.05*sr/hop),1), max(int(0.25*sr/hop),1)
    for i in range(1, len(g_s)):
        w = a_win if g_s[i] < g_s[i-1] else r_win
        g_s[i] = g_s[i-1] + (g_s[i]-g_s[i-1])/w
    gain = np.interp(np.arange(len(y)), np.arange(len(g_s))*hop, g_s)
    return y*(10**(gain/20))

y = expand(y, SR)
y *= 0.89/np.abs(y).max()
sf.write("/tmp/wangye_fupan.wav", y, SR)
fm, fs, dyn, dur = measure(y, SR)
print(f"[终版] F0={fm:.0f} ±{fs:.0f}  dyn={dyn:.0f}dB  dur={dur:.2f}s", flush=True)
subprocess.run(["ffmpeg","-y","-i","/tmp/wangye_fupan.wav","-ar","16000","-ac","1",
                "-c:a","libopus","-b:a","32k","/tmp/wangye_fupan.ogg"],
               check=True, capture_output=True)
print("OGG: /tmp/wangye_fupan.ogg", flush=True)
print("ALL DONE", flush=True)
