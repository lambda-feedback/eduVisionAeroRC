import os
from typing import Any
from lf_toolkit.evaluation import Result, Params
from lf_toolkit.evaluation.image_upload import upload_image, ImageUploadError
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import re
import requests
from requests.exceptions import RequestException
import random
import numpy as np
import cv2

_model_cache = None

def evaluation_function(
    response: Any,
    answer: Any,
    params: Params,
) -> Result:
    """
    Function used to evaluate a student response.
    ---
    The handler function passes three arguments to evaluation_function():

    - `response` which are the answers provided by the student.
    - `answer` which are the correct answers to compare against.
    - `params` which are any extra parameters that may be useful,
        e.g., error tolerances.

    The output of this function is what is returned as the API response
    and therefore must be JSON-encodable. It must also conform to the
    response schema.

    Any standard python library may be used, as well as any package
    available on pip (provided it is added to requirements.txt).

    The way you wish to structure you code (all in this function, or
    split into many) is entirely up to you. All that matters are the
    return types and that evaluation_function() is the main function used
    to output the evaluation response.
    """
    global _model_cache
    # Answer is where the teacher uploaded images as extra data can be loaded from
    print("### Answer: ", answer)
    print("### Response: ", response)
    print("### Params: ", params)
    
    
    # Cache of YOLO model
    if _model_cache is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')
        _model_cache = YOLO(model_path)

    model = _model_cache

    target_class = params.get("target", None)
    print(target_class)
    #if target_class == "":
    #    return Result(
    #        is_correct=False,
    #        feedback_items=[('ERROR', 'No target class specified.')]
    #    )
   



    # Assign a color to each class
    def get_class_color(class_name):
        random.seed(hash(class_name) % 10000)
        return tuple(random.choices(range(50, 256), k=3))

    def draw_annotations_cv2(img, detections, best_idx=None):
        img_cv = np.array(img)
        if img_cv.shape[2] == 4:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        # Draw center dot
        center_x, center_y = w // 2, h // 2
        cv2.circle(img_cv, (center_x, center_y), 7, (0, 255, 255), -1)
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            color = get_class_color(str(cls))
            color_bgr = (int(color[2]), int(color[1]), int(color[0]))
            outline = (0, 0, 255) if i == best_idx else color_bgr
            thickness = 3 if i == best_idx else 2
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), outline, thickness)
            label = f"{cls} : {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            font_thickness = 2
            lbl_margin = 4
            (lbl_w, lbl_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            lbl_w += 2 * lbl_margin
            lbl_h += 2 * lbl_margin
            y1_lbl = max(y1 - lbl_h, 0)
            cv2.rectangle(img_cv, (x1, y1_lbl), (x1 + lbl_w, y1), outline, thickness=-1)
            cv2.putText(img_cv, label, (x1 + lbl_margin, y1 - lbl_margin), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
            # Draw a star in the top-right corner inside the best bbox
            if i == best_idx:
                # Simple 5-point star
                margin = 6
                star_radius = 14
                # Place star center inside the top-right corner
                star_center = (x2 - margin - star_radius, y1 + margin + star_radius)
                pts = []
                for j in range(5):
                    angle = j * 2 * np.pi / 5 - np.pi / 2
                    x = int(star_center[0] + star_radius * np.cos(angle))
                    y = int(star_center[1] + star_radius * np.sin(angle))
                    pts.append((x, y))
                for j in range(5):
                    cv2.line(img_cv, pts[j], pts[(j+2)%5], (0,255,255), 3)
        img_annotated = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_annotated)
    
    def analyze_images(images):
        best_detection = None
        best_conf = 0.0
        analysed_images = 0
        annotated_images = []
        for idx, image in enumerate(images):
            try:
                url = image["url"]
                if url.startswith("file://"):
                    local_path = url[7:]
                    with open(local_path, "rb") as f:
                        img = Image.open(f).convert("RGB")
                else:
                    image_response = requests.get(url)
                    img = Image.open(io.BytesIO(image_response.content)).convert("RGB")
            except Exception as e:
                print('Failed to get image: ', e)
                continue
            results = model.predict(img, conf=0.5)
            detections = []
            fallback_detections = []
            img_width, img_height = img.size
            center_x, center_y = img_width / 2, img_height / 2
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = model.names[int(box.cls[0].item())]
                    # Check if center is inside bbox
                    if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                        detections.append((x1, y1, x2, y2, conf, cls))
                    # Fallback: add all detections
                    fallback_detections.append((x1, y1, x2, y2, conf, cls))
            # Pick best_idx using detections if not empty, else fallback_detections
            used_detections = detections if detections else fallback_detections
            best_idx = None
            if used_detections:
                confs = [d[4] for d in used_detections]
                idx_max = confs.index(max(confs))
                best_det = used_detections[idx_max]
                if best_det[4] > best_conf:
                    best_conf = best_det[4]
                    best_detection = best_det[5]
                # Find the index of the best detection in fallback_detections for annotation
                if detections:
                    # Map best_idx from detections to fallback_detections
                    best_box = used_detections[idx_max][:5]  # (x1, y1, x2, y2, conf)
                    for j, det in enumerate(fallback_detections):
                        if all(np.isclose(det[k], best_box[k]) for k in range(5)):
                            best_idx_fallback = j
                            break
                    else:
                        best_idx_fallback = None
                else:
                    best_idx_fallback = idx_max
            else:
                best_idx_fallback = None
            # Always annotate all fallback_detections
            annotated = draw_annotations_cv2(img.copy(), fallback_detections, best_idx_fallback)
            annotated_images.append((annotated, fallback_detections, best_idx_fallback))
            analysed_images += 1
        return best_conf, best_detection, analysed_images, annotated_images

   

    response_conf, response_detection, analysed_image_count, annotated_images = analyze_images(response)

    if analysed_image_count == 0:
        is_correct = False
        feedback_items = []
        feedback_items.append(('Response', 'Please upload at least one image'))

        return Result(
            is_correct=is_correct,
            feedback_items=feedback_items
        )

    #print(target_class)
    #print(response_detection)
    # Determine if correct based on best detections matching
    is_correct = response_detection == target_class and response_detection is not None
    target_text = f'Target component is {target_class if target_class else "unknown (can be specified in 'target' param)"}.'
    result_text = f'Detected component is {f"{response_detection} ({round(response_conf,2)})" if response_detection else "unknown"}.'


    feedback_items = []
   # Optional parameter: whether to return photos (default True)
    return_images = params.get('return_images', True)
    if return_images:
        # Upload all annotated images (old code below commented out)
        # for idx, (img, detections, best_idx) in enumerate(annotated_images):
        #     try:
        #         feedback_items.append((f'Feedback Image [{idx}]', f'{upload_image(img, "eduvision")} <br>'))
        #     except ImageUploadError as e:
        #         print(f"Failed to upload image feedback {idx}", e)
        for idx, (img, detections, best_idx) in enumerate(annotated_images):
            try:
                img_url = upload_image(img, "eduvision")
                # Extract original filename from the user's uploaded image URL (from response)
                from urllib.parse import urlparse, unquote
                original_url = response[idx]["url"] if idx < len(response) and "url" in response[idx] else None
                if original_url:
                    parsed_orig = urlparse(original_url)
                    orig_filename = os.path.basename(parsed_orig.path)
                    orig_filename = unquote(orig_filename) if orig_filename else f"image_{idx}.jpg"
                else:
                    orig_filename = f"image_{idx}.jpg"
                link_html = f'<a href="{img_url}" target="_blank">{orig_filename}</a>'
                feedback_items.append((f'Feedback Image [{idx}]', link_html))
            except ImageUploadError as e:
                print(f"Failed to upload image feedback {idx}", e)


    show_target = params.get("show_target", True)
    # if show_target somehow came in as a string, convert to bool
    if isinstance(show_target, str):
        show_target = show_target.strip().lower() in ("true", "1", "yes")
    if show_target:
        feedback_items.append(('Target', target_text))
    feedback_items.append(('Result', result_text))

    if params.get('debug', False):
        # print response structure for debugging purposes
        try:
            # use repr to avoid issues with binary data
            feedback_items.append(("DEBUG response structure:", repr(response)))
            print("DEBUG response structure:", repr(response))
        except Exception as e:
            feedback_items.append(("Failed to print response structure", e))
            print("Failed to print response structure", e)

        # also check if YOLO can use GPU (torch.cuda availability)
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = "Error checking GPU availability"
        # sometimes the model itself has a .device attribute
        try:
            model_device = getattr(_model_cache, 'device', None)
            if hasattr(model_device, 'type'):
                model_device = model_device.type
        except Exception:
            model_device = None
        print(f"DEBUG GPU available: {gpu_available}, model.device: {model_device}")
        feedback_items.append(("DEBUG GPU available", f"{gpu_available}, {model_device}"))

        feedback_items.append(('Uploaded Image [0]', f'![Test Image]({response[0]['url']})'))
        feedback_items.append(('Count of Images', f'Image Count: {analysed_image_count}'))

   # If show_target == False, only show Result
    #if not show_target:
    #    feedback_items = [('Result', result_text)]

    return Result(
        is_correct=is_correct,
        feedback_items=feedback_items
    )
