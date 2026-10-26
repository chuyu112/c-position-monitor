"""PTS-aware sequential FFmpeg decode and portable H.264/AAC output."""

import json
import shutil
import subprocess

import numpy as np
import cv2

from .drawing import Overlay, find_font
from .timing import FrameCounter, write_reports


def read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def stop_process(process):
    if process is not None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def render(args, profile, rows, summary):
    from .__main__ import reserve_output

    if not args.video.is_file():
        raise ValueError("Source video does not exist: " + str(args.video))
    binary = shutil.which(args.ffmpeg)
    if not binary:
        raise ValueError("FFmpeg is required; install it or pass --ffmpeg")
    probe_binary = shutil.which(args.ffprobe)
    if not probe_binary:
        raise ValueError("FFprobe is required; install it or pass --ffprobe")
    probe = subprocess.run([probe_binary, "-v", "error", "-show_entries",
                            "stream=codec_type,width,height:format=duration", "-of", "json",
                            str(args.video.resolve())], capture_output=True, text=True, check=False)
    if probe.returncode:
        raise ValueError("Cannot probe source video: " + probe.stderr.strip())
    info = json.loads(probe.stdout)
    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise ValueError("Source contains no video stream")
    source_size = [streams[0]["width"], streams[0]["height"]]
    crop = profile.get("source_crop_xywh")
    if crop:
        x, y, w, h = crop
        if x + w > source_size[0] or y + h > source_size[1]:
            raise ValueError("Source is smaller than the configured crop; supply the original recording")
        if [w, h] != profile["output_size"]:
            raise ValueError("Profile output_size must match the source crop")
    elif source_size != profile["output_size"]:
        raise ValueError("Source dimensions do not match profile: " + str(source_size))
    start = args.start_frame
    if start < 0 or start >= len(rows) or (args.max_frames is not None and args.max_frames <= 0):
        raise ValueError("Invalid render frame range")
    end = min(len(rows), start + args.max_frames) if args.max_frames is not None else len(rows)
    native_width, native_height = profile["output_size"]
    height = args.height if args.height is not None else native_height
    if height < 2 or height % 2:
        raise ValueError("Output height must be a positive even integer")
    width = max(2, round(native_width * height / native_height / 2) * 2)
    overlay = Overlay(profile, width, height, find_font(args.font))
    reserve_output(args.output_dir)
    video = args.output_dir / "annotated.mp4"
    partial = args.output_dir / "annotated.partial.mp4"
    fps = profile["fps"]
    trim = profile.get("source_trim_seconds", 0)
    if float(info.get("format", {}).get("duration", 0)) + 1 / fps < trim + end / fps:
        raise ValueError("Source duration is shorter than the requested annotation timeline")
    filters = []
    if crop:
        x, y, w, h = crop
        filters.append(f"crop={w}:{h}:{x}:{y}")
    filters.extend([f"fps={fps}", f"trim=start_frame={start}:end_frame={end}", "setpts=PTS-STARTPTS",
                    f"scale={width}:{height}:flags=lanczos", "setsar=1"])
    decoder_command = [binary, "-v", "error", "-nostdin", "-ss", str(trim), "-i", str(args.video.resolve()),
                       "-vf", ",".join(filters), "-frames:v", str(end - start), "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    encoder_command = [binary, "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pix_fmt", "bgr24",
                       "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
                       "-ss", str(trim + start / fps), "-i", str(args.video.resolve()),
                       "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                       "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-t", str((end - start) / fps),
                       "-movflags", "+faststart", str(partial)]
    decoder = encoder = None
    try:
        with (args.output_dir / "render.log").open("w", encoding="utf-8") as log:
            decoder = subprocess.Popen(decoder_command, stdout=subprocess.PIPE, stderr=log)
            encoder = subprocess.Popen(encoder_command, stdin=subprocess.PIPE, stderr=log)
            counter = FrameCounter(profile["members"], fps)
            for row in rows[:start]:
                counter.step(row)
            for index in range(start, end):
                raw = read_exact(decoder.stdout, width * height * 3)
                if len(raw) != width * height * 3:
                    raise RuntimeError(f"Video ended before annotation frame {index}; see render.log")
                image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
                counter.step(rows[index])
                overlay.draw(image, rows[index], counter)
                encoder.stdin.write(image.tobytes())
                if index in {start, end - 1}:
                    cv2.imwrite(str(args.output_dir / f"preview-{index:05d}.jpg"), image)
                if (index - start) % 300 == 0:
                    print(json.dumps({"frame": index, "end_frame_exclusive": end}), flush=True)
            decoder.stdout.close()
            encoder.stdin.close()
            if decoder.wait(timeout=60) != 0 or encoder.wait(timeout=60) != 0:
                raise RuntimeError("FFmpeg failed; see render.log")
        partial.replace(video)
    except (BrokenPipeError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("FFmpeg render failed; see render.log") from exc
    finally:
        stop_process(decoder)
        stop_process(encoder)
        if decoder and decoder.stdout:
            decoder.stdout.close()
        if encoder and encoder.stdin:
            try:
                encoder.stdin.close()
            except BrokenPipeError:
                pass
    # Reports describe the full annotation timeline, even when exporting a review clip.
    summary["render"] = {"output": video.name, "start_frame": start, "end_frame_exclusive": end,
                         "frames": end - start, "width": width, "height": height,
                         "source_trim_seconds": trim, "source_crop_xywh": crop,
                         "source_size": source_size, "source_has_audio": any(s.get("codec_type") == "audio" for s in info.get("streams", [])),
                         "counter_origin_frame": 0, "codec": "H.264/AAC", "reports_cover_full_timeline": True}
    write_reports(args.output_dir, profile, rows, summary)
