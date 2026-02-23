from dataclasses import dataclass
from typing import List

import numpy as np
import supervision as sv

@dataclass
class ConfusionSummary:
    matrix: np.ndarray
    tp: int
    fp: int
    fn: int
    tn: int

def compute_confusion_matrix(
    preds_list: List[np.ndarray],
    gts_list: List[np.ndarray],
    class_names: List[str],
    conf_thres: float,
    iou_thres: float
) -> ConfusionSummary:
    cm = sv.ConfusionMatrix.from_tensors(
        predictions=preds_list,
        targets=gts_list,
        classes=class_names,
        conf_threshold=conf_thres,
        iou_threshold=iou_thres,
    )

    m = cm.matrix
    tp = int(m[0, 0])
    fn = int(m[0, 1])
    fp = int(m[1, 0])
    tn = int(m[1, 1])

    return ConfusionSummary(matrix=m, tp=tp, fp=fp, fn=fn, tn=tn)
