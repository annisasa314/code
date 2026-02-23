import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np
from ultralytics import YOLO

@dataclass
class InferenceResult:
    preds_list: List[np.ndarray]  
    gts_list: List[np.ndarray]     
    frame_count: int
    total_time: float
    preprocess_time: float
    inference_time: float
    postprocess_time: float
    avg_fps: float
    output_path: Optional[str] = None

def build_roi_mask(frame_h: int, frame_w: int, roi_points: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    roi_np = np.array(roi_points, dtype=np.int32)
    roi_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv.fillPoly(roi_mask, [roi_np], 255)
    return roi_mask, roi_np

def yolo_txt_to_xyxy(label_path: str, frame_w: int, frame_h: int) -> np.ndarray:
    if not os.path.exists(label_path):
        return np.zeros((0, 5), dtype=np.float32)

    rows = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])

            xc *= frame_w
            yc *= frame_h
            w *= frame_w
            h *= frame_h

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            rows.append([x1, y1, x2, y2, cls])

    if not rows:
        return np.zeros((0, 5), dtype=np.float32)

    arr = np.array(rows, dtype=np.float32)
    arr[:, 0] = np.clip(arr[:, 0], 0, frame_w - 1)
    arr[:, 1] = np.clip(arr[:, 1], 0, frame_h - 1)
    arr[:, 2] = np.clip(arr[:, 2], 0, frame_w - 1)
    arr[:, 3] = np.clip(arr[:, 3], 0, frame_h - 1)
    return arr

def filter_by_roi_xyxy(arr: np.ndarray, roi_poly: np.ndarray) -> np.ndarray:
    if arr is None or len(arr) == 0:
        return arr
    x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    keep = np.zeros((len(arr),), dtype=bool)
    for i in range(len(arr)):
        inside = cv.pointPolygonTest(roi_poly, (float(cx[i]), float(cy[i])), False) >= 0
        keep[i] = inside
    return arr[keep]

def load_model(model_path: str, task: str = "detect") -> YOLO:
    return YOLO(model_path, task=task)

def run_inference_on_video(
    model: YOLO,
    video_path: str,
    roi_points: List[Tuple[int, int]],
    output_path: Optional[str],
    device: str = "cuda",
    conf_thres: float = 0.5,
    iou_thres: float = 0.5,  
    class_names: Optional[List[str]] = None,
    labels_dir: Optional[str] = None,
    label_pattern: str = "frame_{:06d}.txt",
    draw: bool = True
) -> InferenceResult:
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka video: {video_path}")

    frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv.CAP_PROP_FPS)

    roi_mask, roi_np = build_roi_mask(frame_h, frame_w, roi_points)

    out = None
    save_output = False
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        out = cv.VideoWriter(output_path, fourcc, fps_video, (frame_w, frame_h))
        save_output = out.isOpened()

    class_map = {i: class_names[i] for i in range(len(class_names))} if class_names else {0: "Obj"}

    total_time = 0.0
    preprocess_time_total = 0.0
    inference_time_total = 0.0
    postprocess_time_total = 0.0

    preds_list: List[np.ndarray] = []
    gts_list: List[np.ndarray] = []

    frame_count = 0
    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess
            t0 = time.perf_counter()
            frame_masked = cv.bitwise_and(frame, frame, mask=roi_mask)
            t1 = time.perf_counter()
            preprocess_time_total += (t1 - t0)

            # Inference
            t2 = time.perf_counter()
            results = model(frame_masked, verbose=False, device=device)
            r = results[0]
            t3 = time.perf_counter()
            inference_time_total += (t3 - t2)

            pred_frame = np.zeros((0, 6), dtype=np.float32)
            if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                confs = r.boxes.conf.cpu().numpy().astype(np.float32)
                classes = r.boxes.cls.cpu().numpy().astype(np.float32)

                pred_frame = np.column_stack([xyxy, classes, confs]).astype(np.float32)
                pred_frame = pred_frame[pred_frame[:, 5] >= conf_thres]
                pred_frame = filter_by_roi_xyxy(pred_frame, roi_np)

            gt_frame = np.zeros((0, 5), dtype=np.float32)
            if labels_dir is not None:
                label_path = os.path.join(labels_dir, label_pattern.format(frame_idx))
                gt_frame = yolo_txt_to_xyxy(label_path, frame_w, frame_h)
                gt_frame = filter_by_roi_xyxy(gt_frame, roi_np)

            preds_list.append(pred_frame)
            gts_list.append(gt_frame)

            # Postprocess
            t4 = time.perf_counter()
            if draw:
                cv.polylines(frame, [roi_np], True, (0, 0, 255), 3, lineType=cv.LINE_AA)
                if len(pred_frame) > 0:
                    for row in pred_frame:
                        x1, y1, x2, y2, cid, conf = row
                        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                        cid_int = int(cid)
                        cv.rectangle(frame, (x1i, y1i), (x2i, y2i), (255, 0, 0), 2)

                        label_name = class_map.get(cid_int, str(cid_int))
                        label = f"{label_name}: {conf:.2f}"
                        cv.putText(
                            frame, label, (x1i, max(0, y1i - 5)),
                            cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2
                        )
            t5 = time.perf_counter()
            postprocess_time_total += (t5 - t4)

            frame_count += 1
            frame_idx += 1

            if save_output:
                out.write(frame)

    finally:
        cap.release()
        if out is not None:
            out.release()
    
    total_time = preprocess_time_total + inference_time_total + postprocess_time_total
    avg_fps = (frame_count / total_time) if frame_count > 0 else 0.0

    return InferenceResult(
        preds_list=preds_list,
        gts_list=gts_list,
        frame_count=frame_count,
        total_time=total_time,
        preprocess_time=preprocess_time_total,
        inference_time=inference_time_total,
        postprocess_time=postprocess_time_total,
        avg_fps=avg_fps,
        output_path=output_path if save_output else None
    )
