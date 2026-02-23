import os

from input import get_model_configs, get_video_configs, get_roi_config
from inference import load_model, run_inference_on_video
from confusion_matrix import compute_confusion_matrix

def safe_name(s: str) -> str:
    return (
        s.replace(" ", "")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
    )

def main():
    roi_cfg = get_roi_config()
    model_cfgs = get_model_configs()
    video_cfgs = get_video_configs()

    for mcfg in model_cfgs:
        print("=" * 70)
        print(f"Load Model: {mcfg.name}")
        model = load_model(mcfg.path, task=mcfg.task)

        for vcfg in video_cfgs:
            print("-" * 70)
            print(f"Video: {vcfg.speed_label} | {vcfg.input_path}")

            out_name = f"{safe_name(mcfg.name)}__{safe_name(vcfg.speed_label)}.mp4"
            out_path = os.path.join(vcfg.output_dir, out_name)

            result = run_inference_on_video(
                model=model,
                video_path=vcfg.input_path,
                roi_points=roi_cfg.points,
                output_path=out_path,
                device=mcfg.device,
                conf_thres=mcfg.conf_thres,
                iou_thres=mcfg.iou_thres,
                class_names=mcfg.class_names,
                labels_dir=vcfg.labels_dir,
                label_pattern=vcfg.label_pattern,
                draw=True
            )
            print("-" * 70)
            print(f"Model           : {mcfg.name}")
            print(f"Speed           : {vcfg.speed_label}")
            print(f"Frames          : {result.frame_count}")
            print(f"Total           : {result.total_time:.2f}s")
            print(f"Preprocess      : {result.preprocess_time:.2f}s")
            print(f"Inference       : {result.inference_time:.2f}s")
            print(f"Postprocess     : {result.postprocess_time:.2f}s")
            print(f"Mean FPS        : {result.avg_fps:.2f}")
            print(f"Output          : {result.output_path if result.output_path else '(tidak tersimpan)'}")
            print("-" * 70)
            
            if vcfg.labels_dir and mcfg.class_names and len(mcfg.class_names) > 0:
                cm_res = compute_confusion_matrix(
                    preds_list=result.preds_list,
                    gts_list=result.gts_list,
                    class_names=mcfg.class_names,
                    conf_thres=mcfg.conf_thres,
                    iou_thres=mcfg.iou_thres
                )
                print("-" * 70)
                print("Confusion Matrix")
                print(f"True Positive (TP) = {cm_res.tp}")
                print(f"False Positive (FP) = {cm_res.fp}")
                print(f"False Negative (FN) = {cm_res.fn}")
                print(f"True Negative (TN) = {cm_res.tn}")
                print("-" * 70)
            else:
                print("\nlabels_dir atau class_names tidak tersedia -> skip confusion matrix")

    print("Inferensi dan Evaluasi Selesai")

if __name__ == "__main__":
    main()