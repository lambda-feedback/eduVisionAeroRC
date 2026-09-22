# Aero RC Image Evaluation Function — Developer Documentation

This function grades a student's photo submission by running a YOLOv8 object-detection model over each image and checking whether the expected RC component was detected with the highest confidence across the whole response.

> **Only works with the Image Input response type.** The question must be configured to collect images; text/numeric/symbolic responses are not supported.

- Entry point: [`evaluation_function/evaluation.py`](../evaluation_function/evaluation.py) — `evaluation_function(response, answer, params) -> Result`
- Server wiring: [`evaluation_function/main.py`](../evaluation_function/main.py) registers `evaluation_function` as the `eval` handler and `preview_function` as the `preview` handler on the `lf_toolkit` IPC server.
- Local CLI runner: [`evaluation_function/dev.py`](../evaluation_function/dev.py)

## Call Signature

```python
def evaluation_function(response: Any, answer: Any, params: Params) -> Result:
```

This matches `lf_toolkit`'s RPC handler, which calls `_call_user_handler("eval", response, answer, request_params)` — i.e. **`response` is the first positional argument, `answer` is the second**. (`evaluation_function/dev.py` previously called this with the arguments swapped; this has been fixed — see [Known Issues Fixed](#known-issues-fixed-in-this-pass).)

### `response`

**Required** (the only required argument to this function) — a non-empty list of dicts describing the images the student submitted. An empty or falsy `response` (`[]`, `None`, etc.) short-circuits immediately with a "please upload at least one image" feedback message, before `params` is even read or the model is loaded.

Each entry:

```json
[
  { "url": "https://.../photo1.jpg", "name": "photo1.jpg", "size": 123456, "type": "image/jpeg", "comment": "" }
]
```

Only `url` (required) and `name` (optional, used for display) are read. `url` may be an `http(s)://` URL (fetched with `requests.get`) or a `file://` path (used by the local test suite to read a file from disk).

### `answer`

**Not required, and not read anywhere in the function** — whatever the platform passes here is ignored entirely. The expected class comes exclusively from `params.target`. If your question's "answer" field is set in the platform, it has no effect on grading — do not rely on it. This is a deliberate simplification for this function (a single canonical `target` string per question is sufficient for a classification-style task), but it is a sharp edge for anyone used to the "answer vs. response" pattern from other Lambda Feedback functions.

### `params`

See [Configuration Parameters](#configuration-parameters) below. Note that `Params` (from `lf_toolkit.shared.params`) is a `TypedDict` with only `is_latex` / `simplify` / `symbols` declared — but `TypedDict` has no runtime enforcement, so at runtime `params` is just a plain `dict` and `params.get("anything", default)` works for arbitrary keys. That's why this function can read keys like `target`, `model_name`, `draw_images`, etc. that aren't declared anywhere in the type.

## Configuration Parameters

Read with `params.get(name, default)`; **every key is optional** — `params` may be an empty dict and the function still runs (it just won't be gradable without `target`). The only genuinely required input is `response` itself containing at least one image; an empty/missing `response` short-circuits before any of these are even read.

| Parameter | Required? | Type | Default | Effect |
|---|---|---|---|---|
| `target` | Optional | `string` | `None` | Expected class label. Compared with **exact, case-sensitive string equality** against the label of the response's overall best detection. Detection and all feedback generation happen unconditionally regardless of whether `target` is set — omitting it doesn't disable detection, it just means there's nothing to grade against, so `is_correct` is always `False`. This makes `target`-less questions usable as a pure "what did the model detect in my photo?" mode. Set it whenever the question needs an actual pass/fail outcome. |
| `model_name` | Optional | `string` | `"model.pt"` | Which `.pt` weights file (in `evaluation_function/`) to load. Only the basename is used — any directory component is stripped (`os.path.basename`) before joining with the evaluation_function directory, so this can't be used to escape that directory. Models are cached process-wide in `_model_cache`, keyed by the raw (unsanitized) name, so different callers requesting the same string share one loaded model. |
| `conf_threshold` | Optional | `number` | `0.5` | Passed straight through as `model.predict(img, conf=conf_threshold)`. Detections below this confidence are discarded by Ultralytics before this function ever sees them. |
| `allowed_classes` | Optional | `array[string]` | `None` | Restricts detection to only these class names — anything else the model would normally detect is filtered out entirely (not just hidden from feedback). Each name is matched against `model.names` with surrounding whitespace stripped, otherwise exact/case-sensitive, same rules as `target`. Matched names are resolved to their class indices and passed as `model.predict(img, conf=conf_threshold, classes=class_indices)` — Ultralytics itself discards non-matching detections before this function sees them. Names that don't match any class in the loaded model are silently ignored at runtime (no detections for them either way) but are surfaced in `debug` mode feedback (`DEBUG Allowed Classes` block) so typos are easy to catch while building a question. `target` still needs to be one of the allowed names for the question to ever be gradable as correct — this parameter doesn't set `target`, it only narrows what the model looks for. |
| `draw_images` | Optional | `bool` | `True` | Controls both (a) whether bounding boxes are drawn on a copy of each image, and (b) whether that annotated image is uploaded to S3 and embedded in the feedback. When `False`, a plain `---` separator is emitted per image instead. `return_images` is accepted as a **deprecated alias** — if `draw_images` isn't set but `return_images` is, its value is used. This alias exists because an earlier version of this function (and an earlier version of this README) used `return_images` as the primary name; any question still configured that way keeps working. |
| `show_target` | Optional | `bool` | `True` | Whether the "Target component: …" feedback block is emitted. |
| `debug` | Optional | `bool` | `False` | Adds a `DEBUG Times` Markdown table (model load / avg image load / avg prediction / avg postprocess / avg draw / avg upload / analysis / feedback / total, all in seconds) and re-embeds every successfully-annotated image using its **original submitted URL** (not the freshly-uploaded one). Also causes image-upload failures to include the underlying exception message (see [Security Notes](#security-notes)). |
| `debug_response` | Optional | `bool` | `False` | Emits a `DEBUG Response Structure` block containing `repr(response)` — the exact payload the platform handed to this function. Useful for diagnosing malformed/unexpected response shapes. |

## Model Files & Detectable Classes

All three files live in `evaluation_function/` and are loaded via `model_name`. Class names below were read directly from each model's `model.names` (Ultralytics), so they are the authoritative, exact strings required for `target` to match.

### `model.pt` (default) and `model_full_L_Rotation.pt` — 34 classes

Both files expose the identical 34-class label set (`model_full_L_Rotation.pt` was trained with extra rotation-augmented images for more orientation-robust detection; it is otherwise a drop-in replacement).

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

A smaller model scoped to just the wishbone/suspension-arm sub-assembly:

```
suspension, wishbone, front, up,rhs
suspension,wishbone,front,bot,rhs
suspension,wishbone,rear,up,rhs
```

> **Data-quality caveat:** these labels come directly from the training data (`train_data_orginal/*/classes.txt`, git-ignored, local only) and contain inconsistent spacing/punctuation (`suspension, wishbone, front, up,rhs` vs. `suspension,wishbone,front,bot,rhs`) and at least one typo (`veiw Shockabsorber.springnut veiw`, missing the second "i" in "view"). Since matching is exact-string, `target` values must reproduce these quirks verbatim. This is a data/training concern, not something this function's code can safely "fix" — relabeling classes would require retraining the model, since the label strings are baked into the `.pt` file itself.

## Algorithm Walkthrough

For each image in `response`, `analyze_images()` (a closure inside `evaluation_function`) does:

1. **Load.** Fetch the image (`file://` or HTTP). On failure, record a `load_error` for that image and move on — the rest of the response is still processed.
2. **Predict.** `model.predict(img, conf=conf_threshold)` — one YOLOv8 inference pass. Each returned box gives pixel-space `xyxy` corners, a `conf` score, and a `cls` index resolved to a label via `model.names`.
3. **Center-region heuristic.** The image's exact center pixel `(w/2, h/2)` is computed. Detections whose box contains that point go into `det_center`; all detections go into `det_all`. If `det_center` is non-empty, it is used as the candidate pool for this image; otherwise `det_all` is used. This is a deliberate design choice: it assumes the student frames the component roughly in the middle of the photo, and prefers that framed component over incidental background objects the model might also detect.
4. **Per-image best.** Within the candidate pool, the single highest-confidence detection is this image's "best" detection.
5. **Global best.** Across the whole response, if this image's best detection has a higher confidence than the current running maximum, it becomes the response's overall best (`response_detection` / `response_conf`), tracked with which image and which region (center vs. full-image fallback) it came from.
6. **Annotate (optional).** If `draw_images`, every detection is drawn on a copy of the image via OpenCV: a rectangle per detection (color hashed from class name, seeded per-call for determinism), a label (`class: confidence`), the image-center dot, and — only for the image's own best detection — a thicker red outline plus a 5-point star badge in the box's top-right corner.

After all images are processed:

7. **Compare.** `is_correct = (response_detection == target_class) and response_detection is not None`. Exact string equality; no normalization, trimming, or case-folding.
8. **Feedback.** Built via a local `append_feedback(title, text)` helper — see the next section for an important subtlety in how this maps to the platform's `feedback` field.
9. **Upload.** If `draw_images` and the image was annotated, `lf_toolkit.evaluation.image_upload.upload_image` uploads it to the `eduvision` folder of the configured S3 bucket and returns a URL, embedded as a Markdown image.

### Caveat: one bad photo can decide the whole response

Because step 5 tracks a single global maximum confidence across *all* submitted images, a response with several images showing the correct part plus **one** image where the model confidently (mis)detects something else can flip `is_correct` to `False` — even though most of the submitted evidence was correct. This is a real grading-semantics decision baked into the current algorithm (max-confidence-wins, not majority-vote or per-image OR), not a code bug. It is worth being aware of if students report "I submitted the right photo and still got marked wrong" — check the `Overall Best` feedback block (and per-image blocks) to see which image actually won. If this behavior turns out to be undesirable, alternatives worth considering are: requiring the target class to appear as *any* image's best detection (OR-semantics), or a majority vote across images.

## `lf_toolkit.Result` — the tag/body subtlety

`append_feedback(title, text)` builds a tuple `(f"\n\n## {title}\n", f"\n\n{text}\n")` and appends it to `feedback_items`, which is passed to `Result(is_correct=..., feedback_items=feedback_items)`.

**Important:** in `lf_toolkit.evaluation.result.Result`, the first element of each tuple is only used as an internal grouping *tag* (`Result._feedback: Dict[tag, List[body]]`). The public `Result.feedback` property (what actually ends up in the API response) is:

```python
@property
def feedback(self) -> str:
    return "<br>".join(
        feedback_str
        for lists in self._feedback.values()   # iterates VALUES only
        for feedback_str in lists
    )
```

**The tag/title text is never shown to the student.** This is why almost every call site in `evaluation.py` embeds its own `###`-level Markdown heading *inside the body text* (the second argument) rather than relying on the `append_feedback` title — e.g. `append_feedback("Overall Best", "### Best detection across all images\n\n...")`. If you add a new feedback block and want a visible heading, you must put it in the body text, not the title. (`tags` — the list of title keys — is only included in `to_dict()` when called with `include_test_data=True`, which the RPC handler does not do in production.)

## Environment Variables

Required only when `draw_images` is `True` (the default) and at least one image loads successfully, since that's the only path that calls `upload_image`:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `S3_BUCKET_URI` | Yes | — | Base S3 URI annotated images are uploaded under (`<S3_BUCKET_URI>/eduvision/<uuid>.<ext>`). Missing → `MissingEnvironmentVariableError`, caught by this function and reported as an upload failure per image. |
| `AWS_ACCESS_KEY_ID` | Yes | — | Used to SigV4-sign the upload `PUT` request. |
| `AWS_SECRET_ACCESS_KEY` | Yes | — | Same. |
| `AWS_SESSION_TOKEN` | No | — | For temporary/STS credentials. |
| `AWS_REGION` | No | `eu-west-2` | Region used for SigV4 signing. |

`lf_toolkit.evaluation.image_upload` calls `dotenv.load_dotenv()`, so a `.env` file in the working directory is picked up automatically for local development.

## Testing

```bash
poetry install
poetry run pytest --tb=auto -v
```

- `evaluation_function/evaluation_test.py` exercises the happy path (local `file://` image, `debug=True`), the empty-`response` short-circuit, and the deprecated `return_images`/`show_target` alias behavior.
- `evaluation_function/preview_test.py` exercises `preview_function`.
- `evaluation_function/dev.py` provides a CLI runner (`python -m evaluation_function.dev <answer> <response>`) but note it only accepts plain string CLI arguments — it cannot construct the `response` list-of-dicts shape this function actually expects, so it is of limited use for this particular function. Prefer running/extending the pytest tests for local iteration.

CI (`.github/workflows/test-lint.yml`, `staging-deploy.yml`) runs `flake8` (only fatal-error checks are enforced: `E9,F63,F7,F82`) and `pytest` on every PR and on push to `main`.

## Known Issues Fixed in This Pass

The following were found while documenting this function and have been fixed in the code:

1. **Dead/renamed parameters.** The previous README documented `show_target` and `return_images` as live parameters, but the code only read `draw_images` — `return_images` had no effect at all, silently breaking any question configured with it. Both are now honored (`show_target` is wired up again; `return_images` is a supported deprecated alias for `draw_images`).
2. **Bare `except:` clauses** swallowed all exceptions (including e.g. `KeyboardInterrupt`) in two places — image loading and image upload. Both now catch `Exception` specifically and surface useful information: a failed image load is now reported per-image as *"Could not load image: …"* instead of being indistinguishable from *"no component detected"*; a failed upload's underlying error is shown only in `debug` mode (see [Security Notes](#security-notes)).
3. **No feedback for an empty/missing `response`.** Previously this silently produced `is_correct=False` with little to no feedback. Now short-circuits immediately with an explicit "please upload at least one image" message, before even loading the model.
4. **`model_name` path traversal.** `model_name` was joined directly into a filesystem path with no sanitization. It is now reduced to its basename before use, so it can only ever load a file from inside `evaluation_function/`.
5. **`dev.py` argument order bug.** It called `evaluation_function(answer, response, Params())`, but the function's actual positional order is `(response, answer, params)` — the two were being swapped on every CLI invocation. Fixed to `evaluation_function(response, answer, Params())`.
6. **`preview.py` was unmodified boilerplate.** It tried `Preview(sympy=response)`, which is meaningless for an image-list response and would raise inside the `try`, falling into `except Exception as e: return Result(preview=Preview(feedback=str(e)))` — i.e. a raw Python exception string could be shown to a student as "preview feedback". It now always returns an empty `Preview()`, since there is nothing meaningful to preview server-side for an image upload (the platform already shows the uploaded image itself).
7. **Hardcoded confidence threshold.** `model.predict(img, conf=0.5)` is now configurable via the new `conf_threshold` parameter (default unchanged at `0.5`, so existing questions behave identically unless they opt in).
8. Minor: fixed a typo (`Response Structur` → `Response Structure`) in the `debug_response` feedback label.

## Security Notes

- Image-upload failures no longer include the raw exception message in production feedback (only in `debug` mode) — the previous behavior could leak S3/AWS internals (bucket paths, error text from `botocore`) to students.
- `model_name` is sanitized to a basename to prevent path traversal outside `evaluation_function/`. Note this parameter is set by whoever configures the question (an instructor/admin), not by the student directly, so the practical risk was low — but it costs nothing to close off.
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `S3_BUCKET_URI` must be provided via environment variables (or a local `.env`, git-ignored) — never commit them.

## Deployment

Standard Lambda Feedback evaluation-function deployment; see the [boilerplate documentation](https://github.com/lambda-feedback/evaluation-function-boilerplate-python) for the generic Docker/Shimmy/CI mechanics. This repository's specifics:

- `config.json` sets `EvaluationFunctionName` to `eduVisionAeroRC`.
- `.github/workflows/staging-deploy.yml` builds+tests+deploys to staging on every push to `main`.
- `.github/workflows/production-deploy.yml` is a manually-triggered (`workflow_dispatch`) production release.
- Model weight files (`*.pt`, up to ~51 MB) are committed directly to git (no Git LFS — `lfs: false` is set explicitly in the staging deploy workflow) and are copied into the Docker image via `COPY evaluation_function ./evaluation_function` in the `Dockerfile`.
- `train_data_orginal/` is git-ignored and never reaches the deployed image — it's local reference data only, not required at runtime.
