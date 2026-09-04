#!/usr/bin/env bash
# 启动 Mage-VL Video Chat（模型默认为上级目录的本地 checkpoint）
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ~/miniconda3/envs/mage_vl/bin/python app.py --model .. --host 0.0.0.0 --port 8000 "$@"
