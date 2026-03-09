import os
from typing import Any
from lf_toolkit.evaluation import Result, Params
from lf_toolkit.evaluation.image_upload import upload_image
from ultralytics import YOLO
from PIL import Image
import io
import requests
import random
import numpy as np
import cv2
import time



# Cache for loaded models by name
_model_cache = {}


def evaluation_function(
    response: Any,
    answer: Any,
    params: Params,
) -> Result:


    global _model_cache

    print("### Answer: ", answer)
    print("### Response: ", response)
    print("### Params: ", params)

    start_total = time.time()

    draw_images = params.get("draw_images", True)
    model_name = params.get("model_name", "model.pt")

    model_load_start = time.time()

    # Use a dict to cache models by name
    if model_name not in _model_cache:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)
        _model_cache[model_name] = YOLO(model_path)

    model_load_time = time.time() - model_load_start

    model = _model_cache[model_name]

    target_class = params.get("target", None)

    print("Target class:", target_class)

    feedback_items = []

    feedback_start = time.time()

    def append_feedback(title, text):
        feedback_items.append((title + "\n", text.strip() + "\n\n"))

    def get_class_color(class_name):
        random.seed(hash(class_name) % 10000)
        return tuple(random.choices(range(50, 256), k=3))

    def draw_annotations_cv2(img, detections, best_idx=None):

        img_cv = np.array(img)

        if img_cv.shape[2] == 4:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)

        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        h, w = img_cv.shape[:2]
        cx, cy = w // 2, h // 2

        cv2.circle(img_cv, (cx, cy), 7, (0, 255, 255), -1)

        for i, det in enumerate(detections):

            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            color = get_class_color(str(cls))
            color_bgr = (color[2], color[1], color[0])

            outline = (0, 0, 255) if i == best_idx else color_bgr
            thickness = 3 if i == best_idx else 2

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), outline, thickness)

            label = f"{cls}: {conf:.2f}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            cv2.rectangle(img_cv, (x1, y1 - th - 6), (x1 + tw + 10, y1), outline, -1)

            cv2.putText(
                img_cv,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            if i == best_idx:

                margin = 6
                star_r = 14
                star_cx = x2 - margin - star_r
                star_cy = y1 + margin + star_r

                pts = []

                for j in range(5):

                    ang = j * 2 * np.pi / 5 - np.pi / 2

                    px = int(star_cx + star_r * np.cos(ang))
                    py = int(star_cy + star_r * np.sin(ang))

                    pts.append((px, py))

                for j in range(5):
                    cv2.line(img_cv, pts[j], pts[(j + 2) % 5], (0, 255, 255), 3)

        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    def analyze_images(images, draw_images=True):

        best_detection = None
        best_conf = 0.0

        annotated_images = []
        per_image_best = []
        prediction_times = []
        load_times = []
        process_times = []
        draw_times = []

        best_from_center = False
        best_image_idx = None

        for idx, image in enumerate(images):

            load_start = time.time()

            try:

                url = image["url"]

                if url.startswith("file://"):

                    with open(url[7:], "rb") as f:
                        img = Image.open(f).convert("RGB")

                else:

                    img_data = requests.get(url).content
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")

            except:

                load_times.append(time.time() - load_start)

                per_image_best.append(
                    {"best_det": None, "chose_from_center": False}
                )

                annotated_images.append((None, [], None))
                continue

            load_times.append(time.time() - load_start)

            pred_start = time.time()
            results = model.predict(img, conf=0.5)
            prediction_times.append(time.time() - pred_start)

            process_start = time.time()

            det_center = []
            det_all = []

            w, h = img.size
            cx, cy = w / 2, h / 2

            for res in results:

                for box in res.boxes:

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = model.names[int(box.cls[0])]

                    det_all.append((x1, y1, x2, y2, conf, cls))

                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        det_center.append((x1, y1, x2, y2, conf, cls))

            used = det_center if det_center else det_all
            chosen_from_center = bool(det_center)

            best_det = None
            best_idx_fb = None

            if used:

                confs = [d[4] for d in used]
                bi = confs.index(max(confs))
                best_det = used[bi]

                if best_det[4] > best_conf:

                    best_conf = best_det[4]
                    best_detection = best_det[5]
                    best_from_center = chosen_from_center
                    best_image_idx = idx

                for k, d in enumerate(det_all):

                    if all(np.isclose(d[m], best_det[m]) for m in range(5)):

                        best_idx_fb = k
                        break

            process_times.append(time.time() - process_start)

            annotated = None

            if draw_images:
                draw_start = time.time()
                annotated = draw_annotations_cv2(img.copy(), det_all, best_idx_fb)
                draw_times.append(time.time() - draw_start)
            else:
                draw_times.append(0.0)

            annotated_images.append((annotated, det_all, best_idx_fb))

            per_image_best.append(
                {
                    "best_det": best_det,
                    "chose_from_center": chosen_from_center,
                }
            )

        avg_load_time = np.mean(load_times) if load_times else 0.0
        avg_prediction_time = np.mean(prediction_times) if prediction_times else 0.0
        avg_process_time = np.mean(process_times) if process_times else 0.0
        avg_draw_time = np.mean(draw_times) if draw_times else 0.0

        return (
            best_conf,
            best_detection,
            annotated_images,
            per_image_best,
            best_from_center,
            best_image_idx,
            avg_load_time,
            avg_prediction_time,
            avg_process_time,
            avg_draw_time,
        )

    analysis_start = time.time()

    (
        response_conf,
        response_detection,
        annotated_images,
        per_image_best,
        overall_best_from_center,
        overall_best_image_idx,
        avg_load_time,
        avg_prediction_time,
        avg_process_time,
        avg_draw_time,
    ) = analyze_images(response, draw_images)

    analysis_time = time.time() - analysis_start

    if target_class:

        append_feedback(
            "Target",
            f"--- Target ---\n"
            f"Target component: {target_class}",
        )

    if response_detection and len(response) > 1:

        origin = "center region" if overall_best_from_center else "full image"

        overall_name = response[overall_best_image_idx].get(
            "name", f"image_{overall_best_image_idx}.jpg"
        )

        append_feedback(
            "Overall Best",
            f"--- Best detection across all images ---\n"
            f"Image: {overall_name}\n"
            f"Detected Component: {response_detection} ({response_conf:.2f})\n"
            f"Source: {origin}",
        )

    upload_times = []

    for idx, (img, detections, best_idx) in enumerate(annotated_images):

        orig_name = response[idx].get("name", f"image_{idx}.jpg")

        if draw_images and img is not None:

            upload_start = time.time()

            try:

                url = upload_image(img, "eduvision")
                link_html = f'Image: <a href="{url}" target="_blank">{orig_name}\nLink To Annotation: {url}</a>'
                # add separate feedback for this uploaded annotated image
                append_feedback(f"Uploaded Image [{idx}]", f"![{orig_name}]({url})")

            except:

                link_html = f"<b>Image: {orig_name}</b>\nLink To Annotation: (upload failed)"

            upload_times.append(time.time() - upload_start)

        else:

            link_html = f"<b>Image: {orig_name}</b>"
            upload_times.append(0.0)

        info = per_image_best[idx]
        det = info["best_det"]

        if det is None:

            text = "No Component Detected"

        else:

            _, _, _, _, conf, cls = det

            origin = "center region" if info["chose_from_center"] else "full image"

            text = (
                f"Detected Component: {cls}\n"
                f"Confidence: {conf:.2f}\n"
                f"Source: {origin}"
            )

        combined = f"{link_html}\n{text}"

        append_feedback(f"Image [{idx}]", f"--- Image [{idx}] ---\n" + combined)

    avg_upload_time = np.mean(upload_times) if upload_times else 0.0

    feedback_time = time.time() - feedback_start

    total_time = time.time() - start_total

    if params.get('debug', False):
        # print response structure for debugging purposes
        try:
            # use repr to avoid issues with binary data
            append_feedback("DEBUG Response Structure:", f"--- DEBUG Response Structure ---\n{repr(response)}")
            print("DEBUG Response Structure:", repr(response))
        except Exception as e:
            append_feedback("Failed to print response structure", e)
            print("Failed to print response structure", e)

        # also check if YOLO can use GPU (torch.cuda availability)
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = "Error checking GPU availability"
        # sometimes the model itself has a .device attribute
        try:
            model_device = getattr(model, 'device', None)
            if hasattr(model_device, 'type'):
                model_device = model_device.type
        except Exception:
            model_device = None
        print(f"DEBUG GPU Available: {gpu_available}, {model_device}")
        append_feedback("DEBUG GPU Available:", f"--- DEBUG GPU Available ---\n{gpu_available}, {model_device}")

        # include all annotated/uploaded images in debug output
        for idx, (img, _, _) in enumerate(annotated_images):
            if img is not None:
                # we previously uploaded each image and added a feedback entry in the loop above
                # but response urls correspond to originals; to be safe, show the response url here
                name = response[idx].get("name", f"image_{idx}.jpg")
                append_feedback(f'Uploaded Image [{idx}]', f'![{name}]({response[idx]["url"]})')

        append_feedback("DEBUG Times:", f"--- DEBUG Times ---\nModel load: {model_load_time:.3f}s\nAvg image load: {avg_load_time:.3f}s\nAvg prediction: {avg_prediction_time:.3f}s\nAvg detection process: {avg_process_time:.3f}s\nAvg drawing: {avg_draw_time:.3f}s\nAvg upload: {avg_upload_time:.3f}s\nAnalysis: {analysis_time:.3f}s\nFeedback: {feedback_time:.3f}s\nTotal: {total_time:.3f}s")        
    is_correct = response_detection == target_class and response_detection is not None

    return Result(
        is_correct=is_correct,
        feedback_items=feedback_items,
    )
