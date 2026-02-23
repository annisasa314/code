from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class ModelConfig:
    name: str
    path: str
    task: str = "detect"
    device: str = "cuda"   
    conf_thres: float = 0.25
    iou_thres: float = 0.50
    class_names: List[str] = field(default_factory=lambda: ["Lubang"])


@dataclass
class VideoConfig:
    speed_label: str
    input_path: str
    labels_dir: Optional[str] = None
    output_dir: str = "outputs"
    label_pattern: str = "frame_{:06d}.txt"


@dataclass
class ROIConfig:
    points: List[Tuple[int, int]]

def get_roi_config() -> ROIConfig:
    return ROIConfig(points=[
        (538, 0),
        (491, 1077),
        (1331, 1077),
        (1384, 0)
    ])


def get_model_configs() -> List[ModelConfig]:
    return [
        ModelConfig(
            name="yolo10n_engine",
            path="/home/aimp/Documents/icha/semtup/models/yolo10n_HT.engine",
        ),
        ModelConfig(
            name="yolo11n_engine",
            path="/home/aimp/Documents/icha/semtup/models/yolo11n_HT.engine",
        ),
        ModelConfig(
            name="yolo12n_engine",
            path="/home/aimp/Documents/icha/semtup/models/yolo12n_HT.engine",
        ),
    ]


def get_video_configs() -> List[VideoConfig]:
    return [
        VideoConfig(
            speed_label="0.2 m/s",
            input_path="/home/aimp/Documents/icha/semhas/videos/video_0.2.mp4",
            labels_dir="/home/aimp/Documents/icha/semtup/labels/video-0.2",
        ),
        VideoConfig(
            speed_label="0.6 m/s",
            input_path="/home/aimp/Documents/icha/semhas/videos/video_0.6.mp4",
            labels_dir="/home/aimp/Documents/icha/semtup/labels/video-0.6",
        ),
        VideoConfig(
            speed_label="1.0 m/s",
            input_path="/home/aimp/Documents/icha/semhas/videos/video_1.0.mp4",
            labels_dir="/home/aimp/Documents/icha/semtup/labels/video-1.0",
        ),
    ]
