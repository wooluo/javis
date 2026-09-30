# -*- coding: utf-8 -*-
"""王也人设台词 v4：生成+筛优+后处理一条龙（09-30）
人设口吻=慵懒道士；硬基线 F0均值119/±53/动态~42dB(同口径)
配方(实证): 慢锚 wy_long48 + cfg=2.5 + timesteps=40 + 多抽筛优
后处理: atempo 0.92 → 下行扩展 thr=-24/ratio=2.5 → 拼接归一 → 16k opus
"""
import os, sys, time, json, subprocess
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "/home/wooluo/voxemw-deploy/voxemw-app")
import numpy as np, soundfile as sf, librosa

T0 = time.perf_counter()
from voxcpm import VoxCPM
m = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
SR = m.tts_model.sample_rate
print(f"[load] {time.perf_counter()-T0:.1f}s", flush=True)

ANCHOR = "/tmp/wy_long48.wav"
LINES = [
    ("长", "哟，来了？坐吧坐吧。往后想找我说说话，招呼一声就行。我这人您也知道，向来是能躺着绝不坐着。不过朋友来了，还是得起身泡壶茶的。", 3),
    ("短", "想聊什么？修行也好，闲篇儿也罢，风后奇门一开，答案都在里头了。急什么呀，慢慢来。", 4),
]

def measure(y, sr):
    f0,_,_ = librosa.pyin(y=y, fmin=50, fmax=500, sr=sr, frame_length=2048)
    v = f0[~np.isnan(f0)]
    r = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = 20*np.log10(r+1e-9); keep = db > db.max()-60
    d = np.sort(db[keep]); n = max(len(d),2)
    return ((float(v.mean()) if v.size else 0.), (float(v.std()) if v.size else 0.),
            float(d[int(.97*n)]-d[int(.03*n)]), float(len(y)/sr))

cache = m.tts_model.build_prompt_cache(reference_wav_path=ANCHOR)
res, wavs = [], {}
for tag, text, n in LINES:
    for k in range(1, n+1):
        buf = []
        for item in m.tts_model.generate_with_prompt_cache_streaming(
                target_text=text, prompt_cache=cache,
                cfg_value=2.5, inference_timesteps=40):
            w = item[0] if isinstance(item, tuple) else item
            buf.append(np.asarray(w, dtype=np.float32).reshape(-1))
        a = np.concatenate(buf)
        name = f"/tmp/wy4_{tag}_t{k}.wav"
        sf.write(name, a, SR)
        fm, fs, dyn, dur = measure(a.astype(np.float32), SR)
        ok = 90 <= fm <= 160 and fs >= 15
        score = abs(fm-119)/119 + abs(fs-53)/53
        res.append(dict(file=name, tag=tag, f0=round(fm,1), std=round(fs,1),
                        dyn=round(dyn,1), dur=round(dur,2), ok=ok, score=round(score,3)))
        wavs.setdefault(tag, []).append((score if ok else 9e9, name, a))
        print(f"{name}  F0={fm:.0f} ±{fs:.0f}  dyn={dyn:.0f}dB  {dur:.1f}s  "
              f"{'OK' if ok else 'REJ'}", flush=True)

json.dump(res, open("/tmp/wy4_metrics.json","w"), ensure_ascii=False, indent=1)

def atempo_wav(y, sr, rate):
    sf.write("/tmp/_at_in.wav", y, sr)
    subprocess.run(["ffmpeg","-y","-i","/tmp/_at_in.wav","-af",f"atempo={rate}",
                    "/tmp/_at_out.wav"], check=True, capture_output=True)
    a, _ = sf.read("/tmp/_at_out.wav")
    return a.astype(np.float32), sr

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

picked = []
for tag,_,_ in LINES:
    cand = sorted(wavs.get(tag, []))
    if not cand or cand[0][0] >= 9e9:
        print(f"[WARN] {tag} 无合格样张，跳过", flush=True); continue
    _, name, a = cand[0]
    y, sr = atempo_wav(a, SR, 0.92)
    picked.append(expand(y, sr))
    print(f"[PICK] {tag} -> {name}", flush=True)

if picked:
    gap = np.zeros(int(0.8*SR), dtype=np.float32)
    segs = [picked[0]] + [gap] + picked[1:] if len(picked) > 1 else picked
    # 交替拼接
    out = []
    for i, s in enumerate(segs):
        out.append(s)
    final = np.concatenate(out)
    final *= 0.89/np.abs(final).max()
    sf.write("/tmp/wy_persona.wav", final, SR)
    fm, fs, dyn, dur = measure(final, SR)
    print(f"[终版] F0={fm:.0f} ±{fs:.0f} dyn={dyn:.0f}dB dur={dur:.2f}s", flush=True)
    subprocess.run(["ffmpeg","-y","-i","/tmp/wy_persona.wav","-ar","16000","-ac","1",
                    "-c:a","libopus","-b:a","32k","/tmp/wy_persona.ogg"],
                   check=True, capture_output=True)
    print("OGG ready: /tmp/wy_persona.ogg", flush=True)
print("ALL DONE", flush=True)
