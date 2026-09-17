# Aero RC Image Evaluation — Teacher Guide

This evaluation function grades a student's **photo(s)** of an RC vehicle component. A computer-vision model looks at each submitted photo, decides what component it shows, and checks that against the component you specify for the question.

## What students submit

Students upload one or more photos as their response. They don't type anything — there's nothing to compare their answer text against, because grading is entirely based on what the model recognizes in the photo(s).

## Setting up a question

In the question's evaluation function parameters, set:

```json
{
  "target": "shockabsorber_body"
}
```

`target` must be the **exact** component name from the list below (copy-paste it — spelling, spacing, capitalization and punctuation all matter, and a few names have deliberate quirks baked into how the model was trained).

If you don't set `target`, the response will always be marked incorrect — the "answer" field on the question is not used by this function.

### Recommended settings

```json
{
  "target": "shockabsorber_body",
  "show_target": true,
  "draw_images": true,
  "debug": false
}
```

- `show_target: true` — shows students which component they were asked to photograph.
- `draw_images: true` — shows students an annotated copy of their photo (boxes around what was detected), which helps them understand *why* they got the result they did.
- Leave `debug` off for live questions — it's only useful while you're building/testing a question.

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

```json
{
  "target": "suspension,wishbone,front,bot,rhs",
  "model_name": "model_3_PARTS.pt"
}
```

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
