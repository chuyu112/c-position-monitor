"""Shared command line for portrait and landscape reviewed videos."""

import argparse
import json
from pathlib import Path
import sys

from .dataset import load_profile, load_rows
from .timing import summarize, write_reports


def reserve_output(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Output directory must be empty or new: " + str(path))
    path.mkdir(parents=True, exist_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="竖屏与横屏 C 位标注校验、计时和视频导出")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "report", "render"):
        item = sub.add_parser(command)
        item.add_argument("--profile", required=True, help="portrait、landscape 或自定义 JSON 路径")
        item.add_argument("--annotations", type=Path, help="覆盖 profile 中的标注路径")
        item.add_argument("--no-baseline", action="store_true", help="有意修改标注后跳过历史 SHA 和总帧数值核对；仍检查全部数据约束")
        if command != "validate":
            item.add_argument("--output-dir", required=True, type=Path)
        if command == "render":
            item.add_argument("--video", type=Path, required=True, help="原始未画框素材")
            item.add_argument("--ffmpeg", default="ffmpeg")
            item.add_argument("--ffprobe", default="ffprobe")
            item.add_argument("--font", type=Path, help="中文字体；也可设置 C_MONITOR_FONT")
            item.add_argument("--height", type=int, help="等比例缩放输出高度，默认使用 profile 原生尺寸")
            item.add_argument("--start-frame", type=int, default=0, help="短片核对的起始帧，累计时间仍从第 0 帧计算")
            item.add_argument("--max-frames", type=int, help="仅渲染指定帧数，用于核对")
    args = parser.parse_args(argv)
    try:
        profile = load_profile(args.profile)
        rows, digest = load_rows(profile, args.annotations, check_baseline=not args.no_baseline)
        summary = summarize(profile, rows, digest, check_baseline=not args.no_baseline)
        if args.command == "render":
            from .media import render
            render(args, profile, rows, summary)
        elif args.command == "report":
            reserve_output(args.output_dir)
            write_reports(args.output_dir, profile, rows, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(2, "Error: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
