
# Aero RC Image Evaluation Function

This repository contains an evaluation function for the **Aero RC** project on the Lambda Feedback platform. It automatically grades image-based student responses by running a YOLOv8 (Ultralytics) object-detection model over each submitted photo and checking whether the expected RC vehicle component was detected.

> **This evaluation function only works with the Image Input response type.** It must be paired with a question configured to collect images from students; it does not support text, numeric, or symbolic responses.

## Deployment
[![Create Release Request](https://img.shields.io/badge/Create%20Release%20Request-blue?style=for-the-badge)](https://github.com/lambda-feedback/eduVisionAeroRC/issues/new?template=release-request.yml)

## Purpose

- Accept student responses submitted as one or more **images** (photos of a physical RC component).
- Detect and classify components in each image with a pre-trained YOLOv8 model.
- Compare the single most confident detection across the whole response against an expected class, configured per-question via the `target` parameter.
- Return `is_correct` plus rich Markdown feedback: per-image results, annotated/uploaded images, and optional debug/timing information.

**Important:** This function only makes sense for image-upload questions. The `answer` argument passed by the platform is currently **not read at all** — the expected component must be set via `params.target` on the question, not via the question's answer field. See [docs/dev.md](docs/dev.md#the-answer-argument-is-unused) for details.

For the full technical write-up (algorithm, internals, gotchas) see **[docs/dev.md](docs/dev.md)**. For a teacher-facing guide to configuring a question see **[docs/user.md](docs/user.md)**.

## How It Works

1. **Image input:** `response` must be a list of dicts, each with at least a `url` (e.g. `{ "url": "https://...", "name": "photo1.jpg" }`). An empty/missing response short-circuits immediately with feedback asking for an image — no model is loaded.
2. **Model loading:** The requested YOLO model (`model_name`, default `model.pt`) is loaded from the `evaluation_function` directory and cached in memory for the lifetime of the process, so repeated invocations on a warm container are fast.
3. **Detection:** For each image, the model predicts bounding boxes, class labels and confidence scores (`conf_threshold` minimum, default `0.5`). Detections whose box contains the image's center point are preferred; if none do, all detections are considered. Within that pool, the single highest-confidence detection becomes that image's "best" result.
4. **Overall result:** Across *all* submitted images, the single highest-confidence "best" detection (from any one image) becomes the response's overall detected class — this is what gets compared to `target`. See the [multi-image caveat](docs/dev.md#caveat-one-bad-photo-can-decide-the-whole-response) for why this matters.
5. **Annotation:** If enabled, every detected box is drawn on the image (color derived from the class name), the winning "best" box is highlighted with a red outline and a star badge, and the image center is marked with a dot.
6. **Upload & feedback:** Annotated images are uploaded to S3 (via `lf_toolkit`) and embedded in the feedback as Markdown images. Textual feedback reports the target, the per-image result, and the overall best result.
7. **Correctness:** `is_correct` is `True` only if the overall detected class is an **exact, case-sensitive string match** for `target`.

## Parameters

`response` (the student's submitted image(s)) is the only **required** input — if it's empty or missing, the function short-circuits with a "please upload an image" message instead of crashing. Every key in `params` below is **optional**: each is read with `params.get(name, default)`, so a question's configuration can set as few or as many of them as it needs, and any left unset silently fall back to their default.

| Parameter | Required? | Type | Default | Description |
|---|---|---|---|---|
| `target` | Optional | `string` | `None` | Expected component class name. Must exactly match one of the model's class labels (see [class lists](docs/dev.md#model-files--detectable-classes)) for `is_correct` to be `True`. If omitted, detection still runs as normal and all the usual feedback (per-image results, annotated images, overall best) is returned — there's just nothing to grade against, so `is_correct` is always `False`. Useful for detection-only/practice questions where you don't need a pass/fail result. In practice, set this whenever the question needs an actual graded outcome. |
| `model_name` | Optional | `string` | `"model.pt"` | Filename of the `.pt` weights file to load from `evaluation_function/`. Only the filename is used (any path component is stripped), so this cannot be used to load files outside that directory. See available models below. |
| `conf_threshold` | Optional | `number` | `0.5` | Minimum confidence (0–1) Ultralytics requires before returning a detection. |
| `draw_images` | Optional | `bool` | `True` | Whether to draw bounding-box annotations and upload/embed the annotated image in the feedback. `return_images` is accepted as a deprecated alias for backwards compatibility. |
| `show_target` | Optional | `bool` | `True` | Whether to include the "Target component" line in the feedback. |
| `debug` | Optional | `bool` | `False` | Adds a step-by-step timing table and re-embeds every annotated image by its *original submitted* URL. Also includes the underlying error detail when an image upload fails (suppressed otherwise, to avoid leaking S3/AWS internals to students). |
| `debug_response` | Optional | `bool` | `False` | Dumps the raw `repr()` of the entire `response` payload as feedback — useful to see exactly what the platform sent. |

> The `answer` argument and the `is_latex` / `simplify` / `symbols` keys from the generic `Params` type are not used by this function — nothing needs to be set there.

## Detectable Objects / Models

Three model weight files ship in `evaluation_function/`:

| File | Classes | Notes |
|---|---|---|
| `model.pt` (default) | 34 | Full Aero RC drivetrain/suspension component set. |
| `model_full_L_Rotation.pt` | 34 (same set) | Same classes as `model.pt`, trained with additional rotation-augmented images; larger weights file. |
| `model_3_PARTS.pt` | 3 | Only the three wishbone/suspension-arm classes — for questions scoped to that sub-assembly. |

`target` must exactly match one of these labels (case-sensitive, including the spacing/punctuation below) for `is_correct` to be `True`.

### `model.pt` / `model_full_L_Rotation.pt` — 34 classes

```
battery
gearbox bearing
gearbox gear
gearbox sub asse
gearbox shaft+bevel
motor
rear body bracket
rear bracket
DT.f.diff.bevel.ase
DT.f.diff.pinion.ase
Pinion_bearing
pinion
rear diff
rear gear box top
rear suspension tower
shock absorber_top cap
shockabsorber_body
suspension_pivotpin
suspension_wheel
shockabsorber_spring
shockabsorber_uppermount
shock absorber
suspension, wishbone, front, up,rhs
suspension,wishbone,front,bot,rhs
suspension,wishbone,rear,up,rhs
Shaft
Shock absorber.piston-androd
Shockabsorber_rodend
Steering_tierod
Suspension_front_tower
DT.f.diff.main.ase
Shockabsorber.oring
Shockabsorber.sealretainer
veiw Shockabsorber.springnut veiw
```

### `model_3_PARTS.pt` — 3 classes

```
suspension, wishbone, front, up,rhs
suspension,wishbone,front,bot,rhs
suspension,wishbone,rear,up,rhs
```

See [docs/user.md](docs/user.md#full-list-of-valid-target-values) for a teacher-facing version of this list, and [docs/dev.md](docs/dev.md#model-files--detectable-classes) for how these were extracted and the data-quality caveats behind the inconsistent naming.

## Data Format

### Answer (Reference Data)
Not used by this function (see [above](#purpose)).

### Response (Student Submission)
```json
[
  {
    "url": "https://.../photo1.jpg",
    "name": "photo1.jpg",
    "size": 123456,
    "type": "image/jpeg",
    "comment": ""
  }
]
```
`url` may also be a `file://` path (used by the local test suite).

### Params (Evaluation Parameters)
```json
{
  "target": "shockabsorber_body",
  "model_name": "model.pt",
  "conf_threshold": 0.5,
  "draw_images": true,
  "show_target": true,
  "debug": false
}
```

## Output

`Result.to_dict()` (produced by `lf_toolkit`) returns:

| Key | Type | Description |
|---|---|---|
| `is_correct` | `bool` | Whether the overall detected class matches `target`. |
| `feedback` | `string` | All feedback bodies, in order, joined with `<br>`. Contains Markdown (headings, tables, images). |

### Example Output

```json
{
  "is_correct": true,
  "feedback": "\n\n### Target component: shockabsorber_body \n\n---\n\n\n\n### Image photo1.jpg\n\n- **Detected Component:** `shockabsorber_body`\n- **Confidence:** `0.94`\n- **Source:** `center region`\n\n<br>![photo1.jpg](https://.../eduvision/....jpg) \n\n---\n\n"
}
```

## Functionality Details

- **YOLO model:** Uses the Ultralytics YOLOv8 implementation via `ultralytics.YOLO`. Models are cached in-process by filename.
- **Center-region heuristic:** A detection is only preferred over others if its bounding box contains the image's exact center pixel — this assumes students photograph the target component roughly centered in frame.
- **Image annotation:** Every detected box is drawn; the winning detection is outlined in red with a star badge; the image center is marked with a cyan dot.
- **Image upload:** Requires `S3_BUCKET_URI` and AWS credentials to be configured in the environment — see [docs/dev.md](docs/dev.md#environment-variables). If upload fails, feedback shows a short failure note (full error only in `debug` mode).
- **Robustness:** A missing/empty `response` returns clean feedback instead of crashing. A single image that fails to load (bad URL, corrupt file, network error) is reported per-image without affecting the other images.

## Usage Notes

- **Image-only:** This function is designed exclusively for image-upload questions.
- **Exact string matching:** `target` is compared with case-sensitive, whitespace-sensitive string equality against the raw model label — copy class names exactly (including inconsistent spacing/typos baked into the training data, e.g. `veiw Shockabsorber.springnut veiw`). See [docs/user.md](docs/user.md) for the full copy-pasteable list.

## Troubleshooting

- Ensure the chosen `model_name` `.pt` file is present in `evaluation_function/`.
- All required Python packages (see `pyproject.toml`) must be installed.
- If annotated images aren't returned, check `draw_images`.
- If image upload silently fails, set `debug: true` to see the underlying error.
- For debugging, set `debug: true` (timings + re-embedded images) and/or `debug_response: true` (raw payload dump).

## File Structure

- `evaluation_function/evaluation.py` – Main evaluation logic
- `evaluation_function/preview.py` – Preview function (returns an empty preview; there is no meaningful server-side preview for image uploads)
- `evaluation_function/main.py` – IPC server entrypoint
- `evaluation_function/dev.py` – CLI entrypoint for local development (`python -m evaluation_function.dev <answer> <response>`)
- `evaluation_function/evaluation_test.py` / `preview_test.py` – Unit tests
- `evaluation_function/model.pt`, `model_3_PARTS.pt`, `model_full_L_Rotation.pt` – YOLO model weights
- `train_data_orginal/` – Local-only training data (git-ignored, not part of the deployed image)
- `config.json` – Deployment configuration (evaluation function name)
- `docs/dev.md` – In-depth developer documentation
- `docs/user.md` – Teacher-facing documentation

## Contact

For questions or support, contact the Aero RC project maintainers or open an issue in this repository.
