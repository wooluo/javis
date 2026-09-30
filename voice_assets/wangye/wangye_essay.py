# -*- coding: utf-8 -*-
"""王也读散文《匆匆》(朱自清)：慢锚 wy_long48 + cfg2.5 + timesteps40
分块生成→F0/时长校验(崩样重抽)→拼接→atempo 0.92→下行扩展→归一→OGG
"""
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
 ("c1","匆匆，朱自清。燕子去了，有再来的时候；杨柳枯了，有再青的时候；桃花谢了，有再开的时候。但是，聪明的，你告诉我，我们的日子为什么一去不复返呢？是有人偷了他们罢，那是谁？又藏在何处呢？是他们自己逃走了罢，现在又到了哪里呢？"),
 ("c2","我不知道他们给了我多少日子，但我的手确乎是渐渐空虚了。在默默里算着，八千多日子已经从我手中溜去，像针尖上一滴水滴在大海里，我的日子滴在时间的流里，没有声音，也没有影子。我不禁头涔涔而泪潸潸了。"),
 ("c3","去的尽管去了，来的尽管来着；去来的中间，又怎样地匆匆呢？早上我起来的时候，小屋里射进两三方斜斜的太阳。太阳他有脚啊，轻轻悄悄地挪移了，我也茫茫然跟着旋转。"),
 ("c4","于是，洗手的时候，日子从水盆里过去；吃饭的时候，日子从饭碗里过去；默默时，便从凝然的双眼前过去。我觉察他去的匆匆了，伸出手遮挽时，他又从遮挽着的手边过去。"),
 ("c5","天黑时，我躺在床上，他便伶伶俐俐地从我身上跨过，从我脚边飞去了。等我睁开眼和太阳再见，这算又溜走了一日。我掩着面叹息，但是新来的日子的影儿又开始在叹息里闪过了。"),
 ("c6","在逃去如飞的日子里，在千门万户的世界里的我能做些什么呢？只有徘徊罢了，只有匆匆罢了。在八千多日的匆匆里，除徘徊外，又剩些什么呢？"),
 ("c7","过去的日子如轻烟，被微风吹散了，如薄雾，被初阳蒸融了。我留着些什么痕迹呢？我何曾留着像游丝样的痕迹呢？我赤裸裸来到这世界，转眼间也将赤裸裸的回去罢。但不能平的，为什么偏要白白走这一遭啊？"),
 ("c8","你聪明的，告诉我，我们的日子为什么一去不复返呢？"),
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
    for k in range(1, 4):  # 最多3抽
        a = gen(text)
        name = f"/tmp/wye_{tag}_t{k}.wav"
        sf.write(name, a, SR)
        fm, fs, dyn, dur = measure(a, SR)
        dur_ok = (nc/9.0) <= dur <= (nc/1.8)
        ok = (BM-30) <= fm <= (BM+35) and fs >= 15 and dur_ok
        score = abs(fm-BM)/BM + abs(fs-BS)/max(BS,20) + (0 if dur_ok else 1.0)
        res.append(dict(file=name, tag=tag, k=k, f0=round(fm,1), std=round(fs,1),
                        dyn=round(dyn,1), dur=round(dur,2), ok=ok, score=round(score,3)))
        print(f"{name}  F0={fm:.0f} ±{fs:.0f}  dyn={dyn:.0f}dB  {dur:.1f}s({nc}字)  "
              f"{'OK' if ok else 'REJ'}", flush=True)
        if ok and (best is None or score < best[0]):
            best = (score, a)
        if ok: break
    if best is None:
        print(f"[FAIL] {tag} 三抽皆崩，跳段", flush=True); continue
    picked.append(best[1])

json.dump(res, open("/tmp/wye_metrics.json","w"), ensure_ascii=False, indent=1)
if not picked:
    print("NO VALID CHUNKS", flush=True); sys.exit(1)

out = []
for i, y in enumerate(picked):
    if i: out.append(np.zeros(int(0.75*SR), dtype=np.float32))
    out.append(y)
raw = np.concatenate(out)
sf.write("/tmp/wye_raw.wav", raw, SR)
print(f"[拼接] {len(picked)}/{len(CHUNKS)}段  raw={len(raw)/SR:.1f}s", flush=True)

subprocess.run(["ffmpeg","-y","-i","/tmp/wye_raw.wav","-filter:a","atempo=0.92",
                "/tmp/wye_slow.wav"], check=True, capture_output=True)
y,_ = sf.read("/tmp/wye_slow.wav"); y = y.astype(np.float32)

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
sf.write("/tmp/wangye_essay.wav", y, SR)
fm, fs, dyn, dur = measure(y, SR)
print(f"[终版] F0={fm:.0f} ±{fs:.0f}  dyn={dyn:.0f}dB  dur={dur:.2f}s", flush=True)
subprocess.run(["ffmpeg","-y","-i","/tmp/wangye_essay.wav","-ar","16000","-ac","1",
                "-c:a","libopus","-b:a","32k","/tmp/wangye_essay.ogg"],
               check=True, capture_output=True)
print("OGG: /tmp/wangye_essay.ogg", flush=True)
print("ALL DONE", flush=True)
