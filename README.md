
# Aero RC Image Evaluation Function

This repository contains an advanced evaluation function designed for the Aero RC project. Its main purpose is to automatically analyze and evaluate image-based student responses using a YOLO object detection model. The function is tailored for tasks where students upload images as answers, and the evaluation is based on the detection and classification of objects/components within those images.

## Purpose

This evaluation function is intended to:
- Automatically assess student responses that are submitted as images.
- Detect and classify objects/components in the uploaded images using a pre-trained YOLO model.
- Compare detected objects with the expected (target) class and provide detailed, visual feedback.
- Return annotated images, feedback, and correctness information for use in educational platforms.

**Important:** This evaluation function is designed to work exclusively with image inputs. It will not function correctly with text or other data types.

## How It Works

1. **Image Input:** The function expects the `response` argument to be a list of dictionaries, each containing a URL to an image (e.g., `{ "url": "https://..." }`).
2. **Model Loading:** On first use, the function loads a YOLO model from the local `model.pt` file (cached for performance).
3. **Detection:** For each image, the model predicts bounding boxes, classes, and confidence scores. The function prioritizes detections that contain the image center, but will fall back to the most confident detection if needed.
4. **Annotation:** Detected objects are drawn on the images with colored bounding boxes and readable labels. The best detection is highlighted.
5. **Comparison:** The detected class with the highest confidence is compared to the expected target class (provided in parameters).
6. **Feedback:** The function returns:
   - Whether the response is correct (`is_correct`)
   - Annotated feedback images (optional)
   - Textual feedback about the detected and target classes
   - Debug information (optional)

## Parameters

The function accepts the following parameters via the `params` argument:

| Parameter        | Type    | Default | Description |
|------------------|---------|---------|-------------|
| `target`         | string  | None    | The expected class/component to be detected in the image. |
| `show_target`    | bool    | True    | Whether to include information about the target class in the feedback. |
| `return_images`  | bool    | True    | Whether to return annotated feedback images. Set to `False` to omit images from feedback. |
| `debug`          | bool    | False   | If `True`, includes extra debug information in the feedback (such as the original uploaded image URL and image count). |

## Data Format


### Answer (Reference Data)
Not directly used for image analysis, but may contain reference images or metadata for future extensions.

### Params (Evaluation Parameters)
Example:

```json
{
  "target": "servo",
  "show_target": true,
  "return_images": true,
  "debug": false
}
```

## Output

The function returns a dictionary with the following structure:

| Key             | Type    | Description |
|-----------------|---------|-------------|
| `is_correct`    | bool    | Whether the detected class matches the target class. |
| `feedback_items`| list    | List of feedback tuples: (label, value). May include annotated images, target/result info, and debug info. |

### Example Output

```json
{
  "is_correct": true,
  "feedback_items": [
    ["Feedback Image [0]", "<img_url> <br>"],
    ["Target", "Target component is servo."],
    ["Result", "Detected component is servo (0.98)."],
    ["Uploaded Image [0]", "![Test Image](https://example.com/image1.jpg)"],
    ["Count of Images", "Image Count: 1"]
  ]
}
```

## Functionality Details

- **YOLO Model:** Uses the Ultralytics YOLO implementation. The model file (`model.pt`) must be present in the `evaluation_function` directory.
- **Image Annotation:** Bounding boxes and class labels are drawn on the images. The label font size is large for readability. The best detection is highlighted in red.
- **Image Upload:** Annotated images are uploaded and returned as URLs in the feedback (unless `return_images` is set to `False`).
- **Robustness:** If no image is provided, the function returns feedback requesting at least one image.
- **Debug Mode:** If `debug` is enabled, the function includes the original image URL and the number of analyzed images in the feedback.

## Usage Notes

- **Image-Only:** This evaluation function is designed for image-based tasks. Submitting non-image data will result in errors or no evaluation.
- **Aero RC Project:** This function was developed specifically for the Aero RC educational project, but can be adapted for other image-based evaluation scenarios.
- **Extensibility:** The function can be extended to support more advanced feedback, additional parameters, or other object detection models as needed.

## Troubleshooting

- Ensure the `model.pt` file is present in the `evaluation_function` directory.
- All required Python packages (see `pyproject.toml`) must be installed.
- If annotated images are not returned, check the `return_images` parameter.
- For debugging, set `debug` to `true` in the parameters.

## File Structure

- `evaluation_function/evaluation.py` – Main evaluation logic
- `evaluation_function/model.pt` – YOLO model file
- `evaluation_function/preview.py` – Preview utilities
- `evaluation_function/evaluation_test.py` – Tests
- `config.json` – Deployment configuration

## Contact

For questions or support, contact the Aero RC project maintainers or open an issue in this repository.
