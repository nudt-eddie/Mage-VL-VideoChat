# Mage-VL Video Chat

一个基于 Web 的 Mage-VL 演示：打开视频 → **截取单帧**（可框选局部区域，服务端按原生分辨率抽帧）或 **整段/区间视频推理**（codec / frames 后端）→ 输入 prompt 得到回答。

推理复用仓库根目录 `inference.py` 的消息模板与 processor 调用方式（`transformers` offline 模式）。模型常驻显存，首次推理请求时懒加载。

## 启动

```bash
./run.sh                 # 模型默认取上级目录（../），监听 0.0.0.0:8000
# 或手动指定：
conda run -n mage_vl python app.py --model .. --host 0.0.0.0 --port 8000
```

浏览器打开 <http://127.0.0.1:8000>。

依赖：`fastapi`、`uvicorn`、`python-multipart`、`transformers`、`torch`、`PIL`、系统 `ffmpeg`/`ffprobe`（预览转码、抽帧、区间裁剪、元数据）。



![](.\demo.png)

## 功能

| 功能 | 对应 inference.py 路径 | 说明 |
| :--- | :--- | :--- |
| 单帧推理 | `--image`（图片分支） | 播放到目标画面点“截取当前帧”，或直接上传图片；右侧 🖼 单帧推理 |
| 局部框选 | 同上 | 在截取的帧上拖动鼠标框选区域，推理只发送该区域（黄色虚线框） |
| 整段视频推理 | `--video --video-backend codec/frames` | 上传原始文件后服务端处理，后端可选 codec（默认）/frames |
| codec 引擎 | `--codec-engine neural/traditional` | neural = DCVC-RT 神经编解码（默认）；traditional = HEVC |
| 区间推理 | 先 ffmpeg 裁剪再走上述路径 | 填起止秒（或播放到目标点“起点/终点=当前”），只推理该时间段，可“预览区间”核对 |

### 截图 / 预览为什么要走服务端

浏览器 `<video>` 对 AVI、H.265 等编码支持差，且从 `<video>` 元素抓帧会有缩放与转码质量损失。因此：

- **截取单帧**由服务端 ffmpeg 从**原始文件**按精确时间戳抽取**原生分辨率**、高质量（`-q:v 2`）JPEG——无论浏览器能否播放该编码，截图都是原画质。
- **预览**若本地播不了，自动请求 `/api/preview` 用 ffmpeg 转成 H.264 mp4（`crf 23`，仅预览用）；截帧与推理不受其质量影响。
- 文件**只上传一次**，服务端返回 `id`，截帧 / 预览 / 推理全部复用，不重复上传。

### 视频元数据

选择视频后服务端用 ffprobe 展示：分辨率、帧率、总帧数、编码 / 像素格式、时长、码率、容器、文件大小、有无音轨，并实时显示当前推理区间覆盖的帧数。

## API

统一流程：`POST /api/upload` 拿 `id` → 后续接口带 `vid`。

| 方法 & 路径 | 请求 | 返回 |
| :--- | :--- | :--- |
| `GET /` | - | 前端页面 |
| `GET /api/health` | - | `{model_loaded}` |
| `POST /api/upload` | `video`（文件） | `{id, metadata}`，文件保留 30 分钟（闲置自动清理） |
| `DELETE /api/upload/{vid}` | - | 手动删除已上传文件 |
| `GET /api/frame` | `vid`, `t`（秒） | 原生分辨率高质量 JPEG（单帧截图） |
| `POST /api/preview` | `vid` | H.264 mp4（浏览器预览转码） |
| `POST /api/infer` | JSON `{image: dataURL, prompt, max_new_tokens}` | `{answer}`（单帧/裁剪推理） |
| `POST /api/infer_video_file` | 表单 `vid, prompt, backend(codec/frames), codec_engine(neural/traditional), num_frames, max_new_tokens[, start, end]` | `{answer, num_frames, backend, codec_engine, start, end}`（整段或区间推理） |

说明：

- `start`/`end`（秒，可选）只填一个也行（从起点到结尾 / 从头到终点）。
- codec 后端对应 `--video-backend codec`，neural 时 `codec_config.dcvc.pkg_dir` 指向 `<模型目录>/neural_codec`，与 `inference.py` 完全一致。
- 区间裁剪默认 `-c copy` 流拷贝（秒级），容器不支持时自动退化为重编码。

## 文件

- `app.py` — FastAPI 后端（模型加载、上传管理、抽帧、转码、区间裁剪、推理）
- `static/index.html` — 单页前端
- `run.sh` — 启动脚本
- `uploads/` — 服务端临时文件（上传的视频、抽帧/转码产物，用完即删或按 TTL 清理）
