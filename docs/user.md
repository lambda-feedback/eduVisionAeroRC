# Aero RC Image Evaluation — Teacher Guide

This evaluation function grades a student's **photo(s)** of an RC vehicle component. A computer-vision model looks at each submitted photo, decides what component it shows, and checks that against the component you specify for the question.

> **Only works with the Image Input response type.** Set the question up to collect images from students — this function cannot grade text, numeric, or symbolic responses.

## What students submit

Students upload one or more photos as their response. They don't type anything — there's nothing to compare their answer text against, because grading is entirely based on what the model recognizes in the photo(s).

## Setting up a question

1. Add an **Image Input** to the question so students can upload photo(s).
2. Select **this evaluation function** for the question.
3. Under the question's **Evaluation Function Parameters** tab, add key/value pairs as needed — most importantly `target`:

| Key | Value |
|---|---|
| `target` | `shockabsorber_body` |

`target` is **optional**. When set, it must be the **exact** component name from the list below (copy-paste it — spelling, spacing, capitalization and punctuation all matter, and a few names have deliberate quirks baked into how the model was trained).

If you leave `target` unset, the question is never marked correct (there's nothing to grade against) — but detection still runs normally and students still get full feedback on what was detected in their photo(s). This is useful for a practice/exploration question where you just want students to see what the model recognizes, without a pass/fail outcome. Note the "answer" field on the question is not used by this function either way — only `target` drives grading.

### What's required vs. optional

The only thing that's actually required is that the student submits at least one photo — everything else below is optional and falls back to a sensible default if you leave it out.

| Setting | Required? | If you leave it out |
|---|---|---|
| A student photo | **Required** | Nothing to grade — the student sees a "please upload at least one image" message. |
| `target` | Optional | The question can never be marked correct (see above), but detection/feedback still works. Set this whenever you want an actual pass/fail result. |
| `model_name` | Optional | Uses the default full component-set model. Only needed if you want the smaller wishbone-only model. |
| `show_target` | Optional | Defaults to showing the target in feedback. |
| `draw_images` | Optional | Defaults to showing annotated photos in feedback. |
| `debug` / `debug_response` | Optional | Off by default — only turn these on while building/testing a question. |
| `allowed_classes` | Optional | By default the model can detect any component it was trained on. Set this to a list of component names (from the list below) to make it look for only those — useful when a question should only ever recognise a handful of related parts. |

### Recommended settings

| Key | Value |
|---|---|
| `target` | `shockabsorber_body` |
| `show_target` | `true` |
| `draw_images` | `true` |
| `debug` | `false` |

- `show_target: true` — shows students which component they were asked to photograph.
- `draw_images: true` — shows students an annotated copy of their photo (boxes around what was detected), which helps them understand *why* they got the result they did.
- Leave `debug` off for live questions — it's only useful while you're building/testing a question.

### Limiting which components the model looks for (`allowed_classes`)

By default the model will try to recognise any of the components it was trained on, even ones that have nothing to do with the current question. If a question should only ever be graded against a specific subset of parts (e.g. a topic covering just the shock absorber and rear gearbox assembly), set `allowed_classes` to an array of the exact component names — copy them from the list below, same spelling/punctuation rules as `target`:

| Key | Value |
|---|---|
| `allowed_classes` | `["shock absorber", "shockabsorber_spring", "Shockabsorber.oring", "rear diff", "gearbox gear", "gearbox shaft+bevel", "gearbox bearing", "motor", "battery", "gearbox sub asse", "rear gear box top", "Steering_tierod", "suspension, wishbone, front, up,rhs", "suspension,wishbone,front,bot,rhs"]` |

With this set, the model will never report any component outside this list, even if one happens to be visible in the background of a photo. This is separate from `target` — you still need to set `target` (to one of the names in `allowed_classes`) for the question to be gradable as correct/incorrect; `allowed_classes` only narrows what the model is allowed to see.

## Full list of valid `target` values

### Full component set (default model)

Use these with the default model (you don't need to set `model_name` for these):

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

### Wishbone-only set (`model_name: "model_3_PARTS.pt"`)

If your question is specifically about the front/rear wishbone suspension arms, you can point it at the smaller, specialized model:

| Key | Value |
|---|---|
| `target` | `suspension,wishbone,front,bot,rhs` |
| `model_name` | `model_3_PARTS.pt` |

Valid values for this model:

```
suspension, wishbone, front, up,rhs
suspension,wishbone,front,bot,rhs
suspension,wishbone,rear,up,rhs
```

## How a submission is judged

1. The model looks at every photo the student submitted.
2. In each photo, it prefers whatever component is roughly **centered in frame** — so ask students to photograph the part centered, not off to one side.
3. Across all submitted photos, whichever single detection the model is *most confident* about becomes "the" answer that's checked against `target`.

**Practical tip:** because step 3 looks at the single most-confident detection across the *whole* submission, asking students to submit exactly **one clear, centered photo** per question generally gives more predictable results than asking for several. If a question allows multiple photos and a student includes an extra, off-topic photo where the model confidently spots something else, that can outweigh a correct photo elsewhere in the same submission.

## What feedback students see

- The expected component (if `show_target` is on).
- Per photo: what was detected and with what confidence, or a note that nothing was detected / the photo couldn't be loaded.
- If more than one photo was submitted: which photo produced the winning detection.
- If `draw_images` is on: each photo with detected components boxed and labeled, and the winning detection highlighted with a red box and a star.

## Troubleshooting

- **"My student says the photo clearly shows the right part but it was marked wrong."** Check the per-photo feedback (turn on `draw_images` if it's off) — the model may have detected the part with low confidence, or detected something else in the frame with higher confidence, or the part wasn't centered in the photo.
- **Annotated photos aren't showing up in feedback.** Check `draw_images` is `true` for the question.
- **Nothing is being detected at all.** Double check `target` is spelled/punctuated exactly as in the list above, and that `model_name` (if set) matches the model that actually has that class (e.g. the wishbone classes only exist in `model_3_PARTS.pt` as well as the default `model.pt`; most other classes only exist in the default/full model).
- For anything deeper (model internals, parameter reference for developers, known limitations), see **[docs/dev.md](dev.md)**.
