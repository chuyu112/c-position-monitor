"""Run: python -m c_detect /path/to/dance.mp4 --output-dir outputs/demo"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from .core import CenterSelector, Person
from .frontrow import FrontRowSelector
from .overlays import narrow_c_boxes


def parse_args():
    parser = argparse.ArgumentParser(description="固定机位女团视频：队形 C 位候选检测（几何基线）")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--mode", choices=("frontrow", "formation", "relay"), default="frontrow")
    parser.add_argument("--cached-detections", type=Path, help="复用同一视频的 frames.jsonl，重新判断和标注")
    parser.add_argument("--row-tolerance", type=float, default=0.075, help="前排深度容差，相对于人物可见身高")
    parser.add_argument("--axis-x", type=float, default=0.5, help="接力舞舞台中心线，0–1 归一化坐标")
    parser.add_argument("--min-people", type=int, default=None)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--stable-seconds", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=None, help="只处理前 N 帧，便于试跑")
    parser.add_argument("--roi", type=float, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"), help="舞蹈区域，0–1 归一化坐标；按人物躯干位置筛选")
    args = parser.parse_args()
    if args.min_people is None:
        args.min_people = 3 if args.mode == "formation" else 1
    if args.cached_detections and not args.cached_detections.is_file():
        parser.error("找不到缓存检测结果")
    if not args.video.is_file():
        parser.error("找不到输入视频")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("输出目录必须不存在或为空，避免覆盖已有结果")
    if args.imgsz <= 0 or (args.max_frames is not None and args.max_frames <= 0):
        parser.error("imgsz 和 max-frames 必须大于 0")
    if args.roi and not (0 <= args.roi[0] < args.roi[2] <= 1 and 0 <= args.roi[1] < args.roi[3] <= 1):
        parser.error("ROI 应满足 0 <= left < right <= 1，0 <= top < bottom <= 1")
    return args


def extract_people(result, width, height, roi):
    if result.boxes is None or result.boxes.id is None:
        return []
    boxes = result.boxes.xyxy.cpu().tolist()
    ids = result.boxes.id.int().cpu().tolist()
    confidence = result.boxes.conf.cpu().tolist()
    points = result.keypoints.data.cpu().tolist() if result.keypoints is not None else []
    people = []
    for index, (box, track_id, conf) in enumerate(zip(boxes, ids, confidence)):
        normalized = (box[0] / width, box[1] / height, box[2] / width, box[3] / height)
        torso_x, foot_y = None, None
        if index < len(points) and len(points[index]) == 17:
            kpts = points[index]
            visible = lambda k: len(k) >= 3 and k[2] >= 0.5 and 0 < k[0] < width and 0 < k[1] < height
            # Use a complete shoulder or hip pair; one visible side biases x.
            pairs = [(kpts[a][0] + kpts[b][0]) / (2 * width) for a, b in ((5, 6), (11, 12)) if visible(kpts[a]) and visible(kpts[b])]
            if pairs:
                torso_x = sum(pairs) / len(pairs)
            ankles = [kpts[k][1] / height for k in (15, 16) if visible(kpts[k])]
            if ankles:
                foot_y = max(ankles)
        person = Person(track_id, normalized, conf, torso_x, foot_y)
        torso_y = (normalized[1] + normalized[3]) / 2
        if roi and not (roi[0] <= person.x <= roi[2] and roi[1] <= torso_y <= roi[3]):
            continue
        people.append(person)
    return people


def draw_frame(cv2, frame, people, decision, roi):
    height, width = frame.shape[:2]
    selected_people = [p for p in people if p.track_id in decision.selected_ids]
    selected_boxes = dict(zip(
        [p.track_id for p in selected_people],
        narrow_c_boxes([p.box for p in selected_people], width, height,
                       [p.torso_x for p in selected_people]),
    ))
    if roi:
        cv2.rectangle(frame, (int(roi[0] * width), int(roi[1] * height)), (int(roi[2] * width), int(roi[3] * height)), (180, 180, 180), 1)
    if decision.group_x is not None:
        x = int(decision.group_x * width)
        cv2.line(frame, (x, 0), (x, height), (160, 120, 50), 1)
    for person in people:
        selected = person.track_id in decision.selected_ids
        color = (0, 0, 255) if selected else (100, 210, 100)
        x1, y1, x2, y2 = [int(v * (width if i % 2 == 0 else height)) for i, v in enumerate(person.box)]
        if selected:
            x1, y1, x2, y2 = selected_boxes[person.track_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        label = f'ID {person.track_id}{" C?" if selected else ""}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_width = cv2.getTextSize(label, font, 0.55, 1)[0][0]
        scale = max(0.3, min(0.55, 0.55 * max(1, x2 - x1 - 6) / base_width))
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, 1)
        tx = max(2, min(x1 + 3, width - text_width - 3))
        ty = max(42, min(y1 + text_height + 5, height - baseline - 2))
        cv2.rectangle(frame, (tx - 2, ty - text_height - 3), (tx + text_width + 2, ty + baseline), (20, 20, 20), -1)
        cv2.putText(frame, label, (tx, ty), font, scale, color, 1, cv2.LINE_AA)
    if decision.selected_ids:
        label = " + ".join(f"ID {i}" for i in decision.selected_ids)
        text = f'C ({len(decision.selected_ids)}): {label}'
    else:
        text = "C: UNKNOWN"
    cv2.rectangle(frame, (0, 0), (min(width, 640), 32), (20, 20, 20), -1)
    cv2.putText(frame, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


def main():
    args = parse_args()
    selector = (FrontRowSelector(args.stable_seconds, args.row_tolerance) if args.mode == "frontrow"
                else CenterSelector(args.min_people, args.margin, args.stable_seconds, mode=args.mode, axis_x=args.axis_x))
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("请先执行：python -m pip install -r requirements.txt") from exc

    capture = cv2.VideoCapture(str(args.video.resolve()))
    writer = None
    cache = None
    try:
        if not capture.isOpened():
            raise RuntimeError("视频无法打开；请检查格式和文件完整性")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("无法读取视频帧率，请先转成固定帧率 MP4")
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("视频中没有可解码的画面")
        height, width = frame.shape[:2]
        cache_settings = None
        if args.cached_detections:
            cache_summary = json.loads(args.cached_detections.with_name("summary.json").read_text())
            if Path(cache_summary["source"]).resolve() != args.video.resolve() or not math.isclose(cache_summary["fps"], fps, rel_tol=1e-5):
                raise ValueError("缓存不属于同一视频或帧率不一致")
            cache_settings = cache_summary["settings"]
            cache = args.cached_detections.open(encoding="utf-8")
        else:
            model = YOLO(args.model)
            if model.task != "pose":
                raise ValueError("请使用 COCO 17 点人体姿态模型，例如 yolo11n-pose.pt")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.output_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("无法创建 MP4 输出")
        counts, selected_seconds, intervals, cardinality = Counter(), Counter(), [], Counter()
        frame_index = 0
        with (args.output_dir / "frames.jsonl").open("w", encoding="utf-8") as report:
            while ok and (args.max_frames is None or frame_index < args.max_frames):
                timestamp = frame_index / fps
                if cache is not None:
                    line = cache.readline()
                    if not line:
                        raise ValueError("缓存提前结束，未覆盖全部输入帧")
                    cached = json.loads(line)
                    if cached["frame"] != frame_index or abs(cached["seconds"] - timestamp) > 0.001:
                        raise ValueError("缓存帧号或时间与视频不一致")
                    people = [Person(p["track_id"], tuple(p["box"]), p["detection_confidence"], p.get("torso_x"), p.get("foot_y")) for p in cached["people"]]
                else:
                    result = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, imgsz=args.imgsz, device=args.device, verbose=False)[0]
                    people = extract_people(result, width, height, args.roi)
                decision = selector.update(people, timestamp)
                counts[decision.status] += 1
                selected_id = decision.selected_id
                selected_ids = decision.selected_ids
                cardinality[len(selected_ids)] += 1
                for track_id in selected_ids:
                    selected_seconds[track_id] += 1 / fps
                if intervals and intervals[-1]["track_ids"] == selected_ids and intervals[-1]["status"] == decision.status:
                    intervals[-1]["end_seconds"] = (frame_index + 1) / fps
                else:
                    intervals.append({"start_seconds": timestamp, "end_seconds": (frame_index + 1) / fps, "track_id": selected_id, "track_ids": selected_ids[:], "status": decision.status})
                report.write(json.dumps({"frame": frame_index, "seconds": timestamp, "status": decision.status, "selected_id": selected_id, "selected_ids": selected_ids, "c_count": len(selected_ids), "group_x": decision.group_x, "people": [{"track_id": p.track_id, "box": p.box, "torso_x": p.torso_x, "foot_y": p.foot_y, "front_depth_y": decision.depths.get(p.track_id), "detection_confidence": p.confidence, "c_score": decision.scores.get(p.track_id)} for p in people]}, ensure_ascii=False) + "\n")
                writer.write(draw_frame(cv2, frame, people, decision, args.roi))
                frame_index += 1
                if frame_index % 500 == 0:
                    print(f"已处理 {frame_index} 帧 / {frame_index / fps:.1f} 秒", flush=True)
                ok, frame = capture.read()
        summary = {"source": str(args.video.resolve()), "fps": fps, "frames": frame_index, "processed_seconds": frame_index / fps, "settings": {"model": args.model, "device": args.device, "imgsz": args.imgsz, "roi": args.roi, "mode": args.mode, "axis_x": args.axis_x, "min_people": args.min_people, "margin": args.margin, "stable_seconds": args.stable_seconds}, "status_frames": dict(counts), "selected_seconds_by_track": dict(selected_seconds), "intervals": intervals, "notes": ["几何候选分数不是 C 位概率。", "跟踪 ID 不是成员姓名，遮挡换位可能导致 ID 变化。", "输出视频不含音频；时间按固定帧率计算。"]}
        summary["c_count_frames"] = dict(cardinality)
        summary["schema_version"] = 2
        summary["settings"].update({"row_tolerance": args.row_tolerance, "cached_detections": str(args.cached_detections.resolve()) if cache else None, "foot_measurement": "cached_source" if cache else "lower_visible_ankle"})
        if cache_settings:
            summary["settings"]["detector_settings"] = cache_settings
        summary["notes"].append("selected_ids 是正式结果：0–2 个编号。selected_id 仅为兼容字段，双 C 时为 null。每位编号的时长之和属于人秒，双 C 时同一时刻计两人。")
        (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"完成：{args.output_dir.resolve()}（{frame_index} 帧）")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if cache is not None:
            cache.close()


if __name__ == "__main__":
    main()
