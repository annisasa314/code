from ultralytics import YOLO

models = {
    "yolo11n": "models/yolo11n_HT.pt",
}

for name, path in models.items():
    model = YOLO(path)
    model.export(
        format="engine",
        imgsz=640,
        batch=1,
        dynamic=False,
        half=True,
        data="/home/aimp/Documents/icha/data/pothole-10.v2i.yolov11/data.yaml",  
        workspace=4,
    )