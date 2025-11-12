import os
import re
import base64
import io
import sys
from typing import Any
from lf_toolkit.evaluation import Result, Params


# Cache na model i ciężkie biblioteki
_model_cache = None
_lib_cache = {}


def _lazy_import(name: str):
    """
    Lazy import with local cache.
    The first call imports the module, the next one uses the cache.
    """
    if name in _lib_cache:
        return _lib_cache[name]

    if name in sys.modules:
        module = sys.modules[name]
    else:
        import importlib
        module = importlib.import_module(name)

    _lib_cache[name] = module
    return module


def evaluation_function(
    response: Any,
    answer: Any,
    params: Params,
) -> Result:
    """
    Function used to evaluate a student response.
    """

    global _model_cache

    # Lazy import of heavy libraries from cache
    ultralytics = _lazy_import("ultralytics")
    pil = _lazy_import("PIL.Image")

    YOLO = ultralytics.YOLO
    Image = pil.Image

    # Cache of YOLO model
    if _model_cache is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')
        _model_cache = YOLO(model_path)

    model = _model_cache
    target_class = params.get("target", "")

    if target_class == "":
        return Result(
            is_correct=False,
            feedback_items=[('ERROR', 'No target class specified.')]
        )

    def get_best_detection(images):
        best_detection = None
        best_conf = 0.0

        for img_obj in images:
            # Get base64 data
            if isinstance(img_obj, dict) and 'data' in img_obj:
                base64_img = img_obj['data']
            else:
                base64_img = img_obj  # Assume it's a string if not dict

            # Remove prefix and decode
            base64_img = re.sub('^data:.*;base64,', '', base64_img)
            img_data = base64.b64decode(base64_img)
            img = Image.open(io.BytesIO(img_data))

            # YOLO prediction
            results = model.predict(img, conf=0.5)

            img_width, img_height = img.size
            center_x, center_y = img_width / 2, img_height / 2

            valid_detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = model.names[int(box.cls[0].item())]

                    if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                        area = (x2 - x1) * (y2 - y1)
                        valid_detections.append((area, conf, cls))

            if valid_detections:
                valid_detections.sort(reverse=True)
                largest_area, conf, cls = valid_detections[0]

                if conf > best_conf:
                    best_conf = conf
                    best_detection = cls

        return best_detection

    response_detection = get_best_detection(response)
    is_correct = response_detection == target_class and response_detection is not None

    return Result(
        is_correct=is_correct,
        feedback_items=[('Result', 'Target class is ' + target_class + '. \nDetected class is ' + (response_detection if response_detection else 'unknown') + '. ')]
    
    )
