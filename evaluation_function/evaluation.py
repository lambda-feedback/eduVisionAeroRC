import os
from typing import Any
from lf_toolkit.evaluation import Result, Params
from ultralytics import YOLO
import base64
from PIL import Image
import io
import re


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

    print("### Response: ", response)
    print("### Params: ", params)

    model =  YOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt'))

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
            if isinstance(img_obj, dict) and 'data' in img_obj:
                base64_img = img_obj['data']
            else:
                base64_img = img_obj  # Assume it's a string if not dict

            # Decode base64 to image
            base64_img = re.sub('^data:.*;base64,','',base64_img)
            img_data = base64.b64decode(base64_img)
            img = Image.open(io.BytesIO(img_data))

            # Run YOLO prediction
            results = model.predict(img, conf=0.5)  # Adjust conf threshold if needed

            # Get image center
            img_width, img_height = img.size
            center_x, center_y = img_width / 2, img_height / 2

            # Filter detections: bbox must contain center and be the largest
            valid_detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = model.names[int(box.cls[0].item())]

                    # Check if center is inside bbox
                    if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                        area = (x2 - x1) * (y2 - y1)
                        valid_detections.append((area, conf, cls))

            if valid_detections:
                # Sort by area descending, take the largest
                valid_detections.sort(reverse=True)
                largest_area, conf, cls = valid_detections[0]

                # For this class, keep track of highest conf across images
                if conf > best_conf:
                    best_conf = conf
                    best_detection = cls

        return best_detection

   
    response_detection = get_best_detection(response)

    #print(target_class)
    #print(response_detection)
    # Determine if correct based on best detections matching
    is_correct = response_detection == target_class and response_detection is not None

    return Result(
        is_correct=is_correct,
    )
