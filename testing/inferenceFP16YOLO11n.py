import cv2 as cv
import time
import numpy as np
from ultralytics import YOLO

roi_points = [
    (538, 0),
    (491, 1077),
    (1331, 1077),
    (1384, 0)
]
roi_np = np.array(roi_points, dtype=np.int32)


def proses_video(model, video_path, output_path, speed_label):

    print(f"\n=======================================")
    print(f"[INFO] Mulai inferensi video kecepatan {speed_label}")
    print(f"[INFO] Input : {video_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"=======================================\n")

    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[FATAL] Tidak bisa membuka video: {video_path}")
        return

    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv.CAP_PROP_FPS)

    print(f"[INFO] Properti Video → Resolusi: {frame_width}x{frame_height}, FPS: {fps_video:.2f}")

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(output_path, fourcc, fps_video, (frame_width, frame_height))

    if not out.isOpened():
        print("[ERROR] VideoWriter gagal dibuat! Output tidak disimpan.")
        save_output = False
    else:
        save_output = True
        print("[SUCCESS] VideoWriter siap menulis video.")

    roi_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv.fillPoly(roi_mask, [roi_np], 255)

    total_time = 0
    frame_count = 0

    print("[INFO] Mulai memproses frame...\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Semua frame selesai diproses.")
                break

            start_time = time.perf_counter()
            frame_masked = cv.bitwise_and(frame, frame, mask=roi_mask)

            results = model(frame_masked, verbose=False, device="cuda")
            r = results[0]

            end_time = time.perf_counter()

            time_diff = end_time - start_time
            current_fps = 1 / time_diff if time_diff > 0 else 0

            total_time += time_diff
            frame_count += 1

            cv.polylines(frame, [roi_np], True, (0, 0, 255), 3, lineType=cv.LINE_AA)

            if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:

                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)

                box_color = (255, 0, 0)

                class_map = {
                    0: "Lubang:"
                }

                for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, classes):
                    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])

                    cx = int((x1i + x2i) / 2)
                    cy = int((y1i + y2i) / 2)

                    inside = cv.pointPolygonTest(roi_np, (cx, cy), False) >= 0
                    if not inside:
                        continue

                    cv.rectangle(frame, (x1i, y1i), (x2i, y2i), box_color, 2)

                    label_name = class_map.get(cid, str(cid))
                    label = f"{label_name} {conf:.2f}"

                    cv.putText(
                        frame, label, (x1i, y1i - 5),
                        cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2
                    )

            cv.imshow("YOLO11 Detection + ROI", frame)

            if save_output:
                out.write(frame)

            if cv.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Proses dihentikan pengguna.")
                break

    except Exception as e:
        print(f"[FATAL] Error saat proses video: {e}")

    finally:
        cap.release()
        out.release()
        cv.destroyAllWindows()

        if frame_count > 0:
            avg_fps = frame_count / total_time
            print("\n----------------------------------------")
            print(f"[RESULT] Kecepatan : {speed_label}")
            print(f"[RESULT] Total Frame : {frame_count}")
            print(f"[RESULT] Total Waktu : {total_time:.2f} detik")
            print(f"[RESULT] Rata-rata FPS : {avg_fps:.2f}")
            print("----------------------------------------\n")


def main():
    print("\n=======================================")
    print("   INFERENCE YOLO – 3 KECEPATAN VIDEO + ROI")
    print("=======================================\n")

    model_path = "/home/aimp/Documents/icha/models/yolo11n_HT.engine"

    try:
        model = YOLO(model_path, task="detect")
        print(f"[SUCCESS] Model berhasil dimuat: {model_path}")
    except Exception as e:
        print(f"[ERROR] Gagal memuat model: {e}")
        return

    daftar_video = [
        {
            "speed": "0.2 m/s",
            "input": "/home/aimp/Documents/icha/videos/video_0.2.mp4",
            "output": "yolo11_fp16_speed-0.2ms.mp4",
        },
        {
            "speed": "0.6 m/s",
            "input": "/home/aimp/Documents/icha/videos/video_0.6.mp4",
            "output": "yolo11_fp16_speed-0.6ms.mp4",
        },
        {
            "speed": "1.0 m/s",
            "input": "/home/aimp/Documents/icha/videos/video_1.0.mp4",
            "output": "yolo11_fp16_speed-1.0ms.mp4",
        },
    ]

    for vid in daftar_video:
        proses_video(
            model=model,
            video_path=vid["input"],
            output_path=vid["output"],
            speed_label=vid["speed"],
        )

    print("\n===============================")
    print("  SEMUA SKENARIO SELESAI")
    print("===============================\n")


if __name__ == "__main__":
    main()
