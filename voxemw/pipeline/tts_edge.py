# -*- coding: utf-8 -*-
"""edge-tts 云溪 TTS 积木：与 VoxCPM2TTSHandler 同契约接入上游管线。

- process(TTSInput) → 16k int16 mono 音频块 + AUDIO_RESPONSE_DONE
- 按句合成（上游本就按 sentence batch 下发），流式边收边发
- rate/volume/pitch 由 YAML 传参，默认云溪原声
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from threading import Event

import numpy as np

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE

from voxemw.pipeline.launch import strip_stage_directions

logger = logging.getLogger(__name__)

EDGE_TTS_SAMPLE_RATE = 24000
PIPELINE_SAMPLE_RATE = 16000


class EdgeTTSHandler(BaseHandler):
    """edge-tts（微软云溪等神经网络音色）流式 TTS handler。"""

    def setup(
        self,
        should_listen: Event,
        voice: str = "zh-CN-YunxiNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        proxy: str = "",
        blocksize: int = 512,
        cancel_scope=None,
        speculative_turns=None,
        **_unused,
    ) -> None:
        import edge_tts

        self.edge_tts = edge_tts
        self.should_listen = should_listen
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.proxy = proxy or None
        self.blocksize = blocksize
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        logger.info("EdgeTTS 就绪: voice=%s rate=%s pitch=%s", voice, rate, pitch)

    async def _collect_stream(self, text: str):
        """单句合成：流式收 mp3 块，解码为 24k float32 → 16k int16。"""
        import edge_tts

        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch,
            proxy=self.proxy,
        )
        chunks = []
        async for chunk in communicate.stream():
            if self.cancel_scope is not None and self.cancel_scope.discarding:
                break
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)

    def _decode_mp3_to_pcm16k(self, mp3_bytes: bytes) -> np.ndarray:
        """mp3 bytes → 24k float32（soundfile mp3 支持）→ 重采样 16k int16。"""
        import io

        import soundfile as sf
        from scipy.signal import resample_poly

        if not mp3_bytes:
            return np.zeros(0, dtype=np.int16)
        d, sr = sf.read(io.BytesIO(mp3_bytes), dtype="float32")
        if d.ndim > 1:
            d = d.mean(axis=1)
        if sr != PIPELINE_SAMPLE_RATE:
            # 任意采样率 → 管线 16k（edge mp3 为 24k，重采样比 2:3）
            from math import gcd
            g = gcd(int(sr), PIPELINE_SAMPLE_RATE)
            d = resample_poly(d, PIPELINE_SAMPLE_RATE // g, sr // g)
        int16 = np.clip(d * 32768.0, -32768, 32767).astype(np.int16)
        return int16

    def _split_sentences(self, text: str):
        """按句末标点切分；>25 字的长段再按逗号/破折号细分，首句更快出声。"""
        import re
        parts = [p for p in re.split(r"(?<=[。！？!?；;\n])", text) if p.strip()]
        merged = []
        for p in parts:
            if merged and len(p.strip()) < 6:
                merged[-1] += p
            else:
                merged.append(p)
        out = []
        for p in merged:
            if len(p) > 25:
                sub = [s for s in re.split(r"(?<=[，,])", p) if s.strip()]
                buf = ""
                for s in sub:
                    buf += s
                    if len(buf) >= 10:
                        out.append(buf)
                        buf = ""
                if buf:
                    if out and len(buf) < 6:
                        out[-1] += buf
                    else:
                        out.append(buf)
            else:
                out.append(p)
        return out or [text]

    def process(self, tts_input):
        text = strip_stage_directions(getattr(tts_input, "text", "") or "").strip()
        if not text:
            yield AUDIO_RESPONSE_DONE
            return
        import queue as queue_mod
        import threading

        t0 = time.perf_counter()
        first = True
        n_samples = 0
        sentences = self._split_sentences(text)

        q = queue_mod.Queue()  # 无界：打断时 worker 不会卡死在 put
        DONE = object()

        def synth_worker():
            try:
                for sent in sentences:
                    if self.cancel_scope is not None and self.cancel_scope.discarding:
                        break
                    loop = asyncio.new_event_loop()
                    try:
                        mp3 = loop.run_until_complete(self._collect_stream(sent))
                    finally:
                        loop.close()
                    pcm = self._decode_mp3_to_pcm16k(mp3)
                    q.put(pcm)
            except Exception as e:  # noqa: BLE001
                q.put(e)
            finally:
                q.put(DONE)

        th = threading.Thread(target=synth_worker, name="edge-tts-prefetch", daemon=True)
        th.start()

        aborted = False
        while True:
            item = q.get()
            if item is DONE:
                break
            if isinstance(item, Exception):
                logger.warning("EdgeTTS 句子合成失败，跳过该句: %s", item)
                continue
            pcm = item
            if first:
                logger.info("EdgeTTS TTFA(首句): %.2fs", time.perf_counter() - t0)
                first = False
            n_samples += len(pcm)
            for i in collect_step_samples(pcm):
                if self.cancel_scope is not None and self.cancel_scope.discarding:
                    aborted = True
                    break
                yield i
            if aborted:
                break

        if not aborted:
            dur = n_samples / PIPELINE_SAMPLE_RATE
            logger.info("EdgeTTS 生成 %.2fs 音频，耗时 %.2fs（RTF %.2f），共 %d 句",
                        dur, time.perf_counter() - t0,
                        (time.perf_counter() - t0) / max(dur, 1e-6), len(sentences))
        yield AUDIO_RESPONSE_DONE


def collect_step_samples(pcm: np.ndarray, blocksize: int = 512):
    """int16 ndarray → blocksize 字节块迭代器。"""
    raw = pcm.tobytes()
    step = blocksize * 2
    for i in range(0, len(raw), step):
        yield raw[i:i + step]
