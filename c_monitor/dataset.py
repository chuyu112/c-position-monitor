"""Load scene settings and validate reviewed, frame-indexed annotations."""

import hashlib
import json
import math
from pathlib import Path

DATA = Path(__file__).parent / "data"


def load_profile(value):
    path = DATA / "profiles" / (value + ".json") if value in {"portrait", "landscape"} else Path(value)
    path = path.resolve()
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported profile schema_version")
    for key in ("fps", "frame_count", "annotation_size", "output_size", "music_start_frame", "members"):
        if key not in profile:
            raise ValueError("Profile missing " + key)
    if not isinstance(profile["fps"], (int, float)) or not math.isfinite(profile["fps"]) or profile["fps"] <= 0:
        raise ValueError("fps must be finite and positive")
    if type(profile["frame_count"]) is not int or profile["frame_count"] <= 0:
        raise ValueError("frame_count must be a positive integer")
    if type(profile["music_start_frame"]) is not int or not 0 <= profile["music_start_frame"] < profile["frame_count"]:
        raise ValueError("Invalid music_start_frame")
    for key in ("annotation_size", "output_size"):
        size = profile[key]
        if len(size) != 2 or any(type(v) is not int or v < 2 for v in size):
            raise ValueError("Invalid " + key)
    members = profile["members"]
    if not members or any(not isinstance(n, str) or not n for n in members) or len(set(members)) != len(members):
        raise ValueError("members must contain distinct names")
    trim = profile.get("source_trim_seconds", 0)
    if not isinstance(trim, (int, float)) or not math.isfinite(trim) or trim < 0:
        raise ValueError("source_trim_seconds must be nonnegative")
    crop = profile.get("source_crop_xywh")
    if crop is not None and (len(crop) != 4 or any(type(v) is not int for v in crop)
                             or min(crop[:2]) < 0 or min(crop[2:]) <= 0):
        raise ValueError("Invalid source_crop_xywh")
    profile["_path"] = path
    profile["_annotations"] = (path.parent / profile["annotations"]).resolve()
    return profile


def load_rows(profile, annotations=None, check_baseline=True):
    path = Path(annotations) if annotations is not None else profile["_annotations"]
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if check_baseline and profile.get("annotations_sha256") and digest != profile["annotations_sha256"]:
        raise ValueError("Annotations differ from the reviewed baseline; use --no-baseline for intentional revisions")
    rows = []
    width, height = profile["annotation_size"]
    for index, line in enumerate(payload.decode("utf-8").splitlines()):
        if not line.strip():
            raise ValueError("Blank annotation line at frame " + str(index))
        row = json.loads(line)
        frame = row.get("frame")
        if type(frame) is not int or frame != index:
            raise ValueError("Frames must be consecutive integers from zero: " + str(index))
        boxes, names = row.get("boxes"), row.get("members")
        if not isinstance(boxes, list) or not isinstance(names, list) or len(boxes) != len(names) or len(names) > 2:
            raise ValueError("Each frame needs 0–2 boxes with matching member names: " + str(index))
        if any(not isinstance(n, str) or n not in profile["members"] for n in names) or len(set(names)) != len(names):
            raise ValueError("Unknown or duplicate member at frame " + str(index))
        if index < profile["music_start_frame"] and names:
            raise ValueError("Music pre-roll must contain no C boxes: " + str(index))
        for box in boxes:
            if (not isinstance(box, list) or len(box) != 4 or
                    any(type(v) not in (float, int) or not math.isfinite(v) for v in box)):
                raise ValueError("Invalid box coordinates at frame " + str(index))
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height):
                raise ValueError("Box outside annotation canvas at frame " + str(index))
        if "seconds" in row and (not isinstance(row["seconds"], (int, float)) or
                                 not math.isfinite(row["seconds"]) or
                                 abs(row["seconds"] - index / profile["fps"]) > .001):
            raise ValueError("Timestamp does not match frame index: " + str(index))
        rows.append(row)
    if len(rows) != profile["frame_count"]:
        raise ValueError("Annotation count does not match profile")
    return rows, digest
