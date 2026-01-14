import os
from typing import Any
from lf_toolkit.evaluation import Result, Params
from ultralytics import YOLO
from PIL import Image
import io
import re
import requests
from requests.exceptions import RequestException

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
   

    def get_best_detection(images):
        best_detection = None
        best_conf = 0.0
        analysed_images = 0

        for image_url in images:
            try:
                image_response = requests.get(image_url)
                img = Image.open(io.BytesIO(image_response.content))
            except RequestException as e:
                print('Failed to get image: ', e)
                continue

            # Run YOLO prediction
            results = model.predict(img, conf=0.5)  # Adjust conf threshold if needed

            # Get image center
            img_width, img_height = img.size
            center_x, center_y = img_width / 2, img_height / 2

            # Filter detections: bbox must contain center and be the largest
            valid_detections = []
            fallback_detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = model.names[int(box.cls[0].item())]
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Check if center is inside bbox
                    if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                        valid_detections.append((conf, area, cls))
                    # Add to fallback
                    fallback_detections.append((conf, area, cls))

            if valid_detections:
                # Sort by conf descending, take the largest
                valid_detections.sort(reverse=True)
                conf, largest_area, cls = valid_detections[0]

                # For this class, keep track of highest conf across images
                if conf > best_conf:
                    best_conf = conf
                    best_detection = cls
            
            elif fallback_detections:
                # Sort by conf descending, take the largest
                fallback_detections.sort(reverse=True)
                conf, largest_area, cls = fallback_detections[0]

                # For this class, keep track of highest conf across images
                if conf > best_conf:
                    best_conf = conf
                    best_detection = cls

            analysed_images += 1


        return best_conf, best_detection, analysed_images

   
    response_conf, response_detection, analysed_image_count = get_best_detection(response)

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
    show_target = params.get("show_target", True)
    if show_target:
        feedback_items.append(('Target', target_text))
    feedback_items.append(('Result', result_text))

    # Jeśli show_target == False, pokazuj tylko Result
    if not show_target:
        feedback_items = [('Result', result_text)]

    return Result(
        is_correct=is_correct,
        feedback_items=feedback_items
    )
