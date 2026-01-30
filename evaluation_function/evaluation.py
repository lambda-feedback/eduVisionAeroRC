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

    def draw_annotations(img, detections, best_idx=None):
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det
            color = get_class_color(cls)
            width = 4 if i == best_idx else 2
            outline = color if i != best_idx else (255, 0, 0)
            draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)
            label = f"{cls} {conf:.2f}"
            text_size = draw.textsize(label, font=font)
            draw.rectangle([x1, y1 - text_size[1], x1 + text_size[0], y1], fill=outline)
            draw.text((x1, y1 - text_size[1]), label, fill=(255,255,255), font=font)
        return img

    def analyze_images(images):
        best_detection = None
        best_conf = 0.0
        analysed_images = 0
        annotated_images = []
        for idx, image in enumerate(images):
            try:
                image_response = requests.get(image["url"])
                img = Image.open(io.BytesIO(image_response.content)).convert("RGB")
            except RequestException as e:
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
            # Use detections containing center if any, else fallback to all
            used_detections = detections if detections else fallback_detections
            best_idx = None
            if used_detections:
                confs = [d[4] for d in used_detections]
                idx_max = confs.index(max(confs))
                best_det = used_detections[idx_max]
                if best_det[4] > best_conf:
                    best_conf = best_det[4]
                    best_detection = best_det[5]
                best_idx = idx_max
            annotated = draw_annotations(img.copy(), used_detections, best_idx)
            annotated_images.append((annotated, used_detections, best_idx))
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
    # Upload all annotated images
    for idx, (img, detections, best_idx) in enumerate(annotated_images):
        try:
            feedback_items.append((f'Feedback Image [{idx}]', f'{upload_image(img, "eduvision")} <br>'))
        except ImageUploadError as e:
            print(f"Failed to upload image feedback {idx}", e)


    show_target = params.get("show_target", True)
    if show_target:
        feedback_items.append(('Target', target_text))
    feedback_items.append(('Result', result_text))

    if params.get('debug', False):
        feedback_items.append(('Uploaded Image [0]', f'![Test Image]({response[0]['url']})'))
        feedback_items.append(('Count of Images', f'Image Count: {analysed_image_count}'))

    # Jeśli show_target == False, pokazuj tylko Result
    if not show_target:
        feedback_items = [('Result', result_text)]

    return Result(
        is_correct=is_correct,
        feedback_items=feedback_items
    )
