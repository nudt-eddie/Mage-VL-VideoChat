#!/usr/bin/env python
"""Web demo for Mage-VL: open a video, capture/crop a frame, ask a question.

Mirrors the offline image path of inference.py, but keeps the model resident.
Run:
    conda run -n mage_vl python app.py            # model = parent directory
    conda run -n mage_vl python app.py --model /path/to/Mage-VL

流程：选择视频时上传一次拿 id（/api/upload 返回元数据），之后截帧（/api/frame，
服务端 ffmpeg 从原始文件抽原生分辨率帧）、预览转码（/api/preview）、区间/整段推理
（/api/infer_video_file）都复用同一个文件，不重复上传。
"""

import argparse
import base64
import io
import os
import subprocess
import threading
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

MODEL_DIR = None  # set in main()
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
UPLOAD_TTL = 30 * 60  # 已上传文件闲置 30 分钟后清理

app = FastAPI(title="Mage-VL Video Chat")

_state = {"processor": None, "model": None}
_load_lock = threading.Lock()

# id -> {"path": str, "ts": float}，后台线程按 TTL 清理
_uploads = {}
_uploads_lock = threading.Lock()


def _sweeper():
    while True:
        time.sleep(60)
        now = time.time()
        with _uploads_lock:
            stale = [vid for vid, rec in _uploads.items() if now - rec["ts"] > UPLOAD_TTL]
            for vid in stale:
                rec = _uploads.pop(vid)
                try:
                    os.remove(rec["path"])
                except OSError:
                    pass
        if stale:
            print(f"[uploads] TTL 清理 {len(stale)} 个文件", flush=True)


def load_model():
    """Lazy-load model/processor on first inference (startup stays fast)."""
    with _load_lock:
        if _state["model"] is not None:
            return
        import transformers
        import transformers.dynamic_module_utils

        transformers.dynamic_module_utils.resolve_trust_remote_code = lambda *a, **k: True

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f"[Mage-VL] loading model from {MODEL_DIR} ...", flush=True)
        _state["processor"] = AutoProcessor.from_pretrained(
            MODEL_DIR, trust_remote_code=True
        )
        _state["model"] = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_DIR, trust_remote_code=True, torch_dtype="auto", device_map="auto"
            ).eval()
        )
        print("[Mage-VL] model ready", flush=True)


def decode_data_url(url: str):
    """data URL 或纯 base64 -> PIL RGB image"""
    from PIL import Image

    payload = url.split(",", 1)[1] if url.startswith("data:") else url
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def _run(cmd, timeout=600):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _probe(path: str) -> dict:
    """ffprobe 读取视频元数据：分辨率、帧率、总帧数、编码、时长、码率、大小。"""
    import json

    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    proc = _run(cmd, timeout=60)
    if proc.returncode != 0:
        raise HTTPException(422, f"ffprobe 失败: {proc.stderr[-300:]}")
    probe = json.loads(proc.stdout)

    vstream = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None
    )
    if vstream is None:
        raise HTTPException(422, "文件中没有视频流")
    fmt = probe.get("format", {})

    def _try(fn, *a):
        try:
            return fn(*a)
        except (ValueError, TypeError, KeyError, ZeroDivisionError):
            return None

    # 帧率/总帧数：nb_frames 不可靠（AVI 等常缺失），优先用 r_frame_rate × duration
    num, den = (vstream.get("r_frame_rate") or "0/1").split("/")
    fps = _try(lambda: int(num) / int(den))
    duration = _try(float, vstream.get("duration")) or _try(float, fmt.get("duration"))
    if not fps and duration and vstream.get("avg_frame_rate") not in (None, "0/0"):
        num2, den2 = vstream["avg_frame_rate"].split("/")
        fps = _try(lambda: int(num2) / int(den2))
    total_frames = _try(int, vstream.get("nb_frames"))
    if not total_frames and fps and duration:
        total_frames = round(fps * duration)

    audio = next((s for s in probe.get("streams", []) if s.get("codec_type") == "audio"), None)

    return {
        "filename": os.path.basename(path),
        "size_bytes": int(fmt.get("size") or 0),
        "container": fmt.get("format_name"),
        "duration": duration,
        "width": vstream.get("width"),
        "height": vstream.get("height"),
        "codec": vstream.get("codec_name"),
        "codec_long": vstream.get("codec_long_name"),
        "pix_fmt": vstream.get("pix_fmt"),
        "fps": round(fps, 3) if fps else None,
        "total_frames": total_frames,
        "bitrate": _try(int, fmt.get("bit_rate")),  # bps, 总码率（含音频）
        "has_audio": audio is not None,
        "audio_codec": audio.get("codec_name") if audio else None,
    }


def _get_upload(vid: str) -> dict:
    with _uploads_lock:
        rec = _uploads.get(vid)
    if rec is None:
        raise HTTPException(404, "视频未上传或已过期，请重新选择文件")
    rec["ts"] = time.time()  # 续期
    return rec


class InferRequest(BaseModel):
    image: str = Field(..., description="data URL or base64 JPEG/PNG of the frame/crop")
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = 256


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/health")
def health():
    return {"model_loaded": _state["model"] is not None}


@app.post("/api/upload")
def upload(video: UploadFile = File(...)):
    """上传视频一次：保存 + ffprobe 元数据，返回 id 供后续接口复用。"""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    vid = uuid.uuid4().hex
    path = os.path.join(UPLOAD_DIR, f"{vid}{suffix}")
    with open(path, "wb") as f:
        f.write(video.file.read())
    if os.path.getsize(path) == 0:
        try:
            os.remove(path)
        except OSError:
            pass
        raise HTTPException(400, "上传的视频文件为空")
    try:
        meta = _probe(path)
    except HTTPException:
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    meta["filename"] = video.filename
    with _uploads_lock:
        _uploads[vid] = {"path": path, "ts": time.time()}
    print(f"[upload] {video.filename!r} -> {vid} ({meta['width']}x{meta['height']})", flush=True)
    return {"id": vid, "metadata": meta}


@app.delete("/api/upload/{vid}")
def delete_upload(vid: str):
    with _uploads_lock:
        rec = _uploads.pop(vid, None)
    if rec:
        try:
            os.remove(rec["path"])
        except OSError:
            pass
    return {"deleted": rec is not None}


@app.get("/api/frame")
def extract_frame(vid: str, t: float = 0.0):
    """从已上传的原始视频抽取 t 秒处的帧——原生分辨率、服务端 ffmpeg 解码。

    不经过浏览器 <video>（避免其编解码缩放/预览转码的质量损失）。
    """
    rec = _get_upload(vid)
    src = rec["path"]
    dst = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.jpg")
    # -ss 放 -i 前（快速 seek），-q:v 2 高质量 JPEG，分辨率不动
    cmd = [
        "ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", src,
        "-frames:v", "1", "-q:v", "2", "-loglevel", "error", dst,
    ]
    proc = _run(cmd, timeout=120)
    if proc.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        try:
            os.remove(dst)
        except OSError:
            pass
        raise HTTPException(422, f"抽帧失败: {proc.stderr[-300:]}")
    return FileResponse(dst, media_type="image/jpeg", background=_delete_later(dst))


def _delete_later(path: str):
    """FileResponse 发送完成后再删除临时转码文件。"""
    from starlette.background import BackgroundTask

    return BackgroundTask(lambda p=path: os.path.exists(p) and os.remove(p))


def _cut_video(src: str, start: float, end: float) -> str:
    """用 ffmpeg 截取 [start, end] 秒区间到新临时文件，返回其路径。

    不重编码（-c copy），只是流拷贝，秒级完成。关键帧对齐误差由调用方接受
    （浏览器里预览的也是同一文件，所见即推理范围）。
    """
    end = min(end, 1e9)  # float("inf") 格式化成字符串 ffmpeg 不认
    dst = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", src,
        "-c", "copy", "-loglevel", "error", dst,
    ]
    proc = _run(cmd, timeout=120)
    if proc.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        # 流拷贝对某些容器（如 AVI/MJPEG）会失败，退化为重编码
        dst2 = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.mp4")
        cmd2 = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", src,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-an", "-movflags", "faststart",
            "-loglevel", "error", dst2,
        ]
        proc2 = _run(cmd2, timeout=600)
        for p in (dst, dst2):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        if proc2.returncode != 0:
            raise HTTPException(422, f"ffmpeg 截取失败: {proc2.stderr[-300:]}")
        return dst2
    return dst


@app.post("/api/preview")
def preview(vid: str = Form(...)):
    """把浏览器播不了的视频（如部分 AVI 编码）转码成 H.264 mp4 用于前端预览。

    只影响预览播放；截帧（/api/frame）与推理（/api/infer_video_file）都
    直接用已上传的原始文件，不受此转码质量影响。
    """
    rec = _get_upload(vid)
    src = rec["path"]
    dst = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.mp4")
    # -movflags faststart: moov 前置，浏览器可边下边播；分辨率不变
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-movflags", "faststart",
        "-loglevel", "error",
        dst,
    ]
    proc = _run(cmd, timeout=600)
    if proc.returncode != 0:
        try:
            os.remove(dst)
        except OSError:
            pass
        raise HTTPException(422, f"ffmpeg 转码失败: {proc.stderr[-500:]}")
    return FileResponse(dst, media_type="video/mp4", background=_delete_later(dst))


@app.post("/api/infer")
def infer(req: InferRequest):
    import torch

    load_model()
    processor, model = _state["processor"], _state["model"]

    image = decode_data_url(req.image)

    # Same message format as inference.py run_offline
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": req.prompt}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=req.max_new_tokens, do_sample=False
        )
    answer = processor.tokenizer.decode(
        output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return {"answer": answer.strip()}


@app.post("/api/infer_video_file")
def infer_video_file(
    vid: str = Form(...),
    prompt: str = Form(...),
    backend: str = Form("codec"),
    codec_engine: str = Form("neural"),
    num_frames: int = Form(32),
    max_pixels: int = Form(150000),
    max_new_tokens: int = Form(256),
    start: float = Form(None),
    end: float = Form(None),
):
    """整段或区间视频推理：复用已上传的原始视频（/api/upload 返回的 id）。

    start/end（秒，可选）：给出时先用 ffmpeg 截取该区间再推理。

    backend:
      - "codec"（默认）: --video-backend codec，codec_engine 可选
          "neural"（DCVC-RT 神经编解码，--codec-engine neural）
          "traditional"（HEVC，--codec-engine traditional）
      - "frames": --video-backend frames，均匀采样
    """
    import torch

    if backend not in ("codec", "frames"):
        raise HTTPException(400, "backend 只支持 codec / frames")
    if codec_engine not in ("neural", "traditional"):
        raise HTTPException(400, "codec_engine 只支持 neural / traditional")
    if backend == "frames" and codec_engine != "traditional":
        # frames 后端不区分引擎，规范化一下避免歧义
        codec_engine = "traditional"

    load_model()
    processor, model = _state["processor"], _state["model"]

    rec = _get_upload(vid)
    path = rec["path"]
    infer_path = path  # 实际送入 processor 的文件（可能被截取过）
    cut_tmp = None
    try:
        if start is not None or end is not None:
            cut_tmp = infer_path = _cut_video(path, start or 0.0, end or float("inf"))

        messages = [
            {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if backend == "codec":
            # 与 inference.py run_offline 的 codec 分支完全一致
            codec_config = {
                "engine": "dcvc-rt" if codec_engine == "neural" else "hevc",
                "target_canvas": num_frames,
                "patch": 16,
            }
            if codec_engine == "neural":
                codec_config["dcvc"] = {
                    "pkg_dir": os.path.join(MODEL_DIR, "neural_codec"),
                    "device": str(model.device),
                }
            inputs = processor(
                text=[text],
                videos=[infer_path],
                video_backend="codec",
                max_pixels=max_pixels,
                codec_config=codec_config,
                return_tensors="pt",
                trust_remote_code=True,
                padding=True,
            )
        else:
            inputs = processor(
                text=[text],
                videos=[infer_path],
                num_frames=num_frames,
                return_tensors="pt",
                padding=True,
            )
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        answer = processor.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        cut_desc = f" cut={start}-{end}s" if (start is not None or end is not None) else ""
        print(
            f"[infer_video_file] vid={vid[:8]}… backend={backend}/{codec_engine}{cut_desc} "
            f"num_frames={num_frames} -> {answer[:200]!r}",
            flush=True,
        )
        return {
            "answer": answer,
            "num_frames": num_frames,
            "backend": backend,
            "codec_engine": codec_engine,
            "start": start,
            "end": end,
        }
    finally:
        if cut_tmp and cut_tmp != path:
            try:
                os.remove(cut_tmp)
            except OSError:
                pass


def main():
    global MODEL_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="..", help="Path to the Mage-VL checkpoint (default: parent dir)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    MODEL_DIR = os.path.abspath(os.path.expanduser(args.model))

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    threading.Thread(target=_sweeper, daemon=True).start()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
