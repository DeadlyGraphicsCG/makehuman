# Modifier Mapping — Loomis ratio ↔ MakeHuman target

The analyzer computes 10 dimensionless ratios in `W_eye = 1.0` units. Step C
maps each onto one MakeHuman modifier path and resolves it to a concrete
`.target` file based on the sign of the value.

## Ratio definitions

All distances measured in eye-line-aligned coordinates after roll-correction.
`W_eye` (eye-width) is the average of the two eye apertures (outer to inner
canthus per eye).

| Ratio key | Formula | Loomis baseline |
|-----------|---------|-----------------|
| `face_width_per_eye` | `eye_l_outer.x − eye_r_outer.x` ÷ W_eye | **5.0** (the canonical "5 eye widths across") |
| `face_height_per_eye` | `forehead_top.y − chin.y` ÷ W_eye | **7.0** |
| `intereye_per_eye` | `eye_l_inner.x − eye_r_inner.x` ÷ W_eye | **1.0** (one eye fits between inner canthi) |
| `ipd_per_eye` | `iris_l_center → iris_r_center` ÷ W_eye | **2.0** |
| `nose_width_per_eye` | `nose_wing_l.x − nose_wing_r.x` ÷ W_eye | **1.0** |
| `mouth_width_per_eye` | `mouth_corner_l.x − mouth_corner_r.x` ÷ W_eye | **1.5** |
| `hairline_to_brow_per_eye` | `forehead_top.y − brow_center.y` ÷ W_eye | **2.0** (top third of face) |
| `brow_to_nose_per_eye` | `brow_center.y − nose_bottom.y` ÷ W_eye | **2.0** (middle third) |
| `nose_to_chin_per_eye` | `nose_bottom.y − chin.y` ÷ W_eye | **2.0** (bottom third) |
| `eye_to_mouth_per_eye` | `eye_l_inner.y − mouth_upper.y` ÷ W_eye | **2.5** |

## MediaPipe landmark indices used

| Landmark name | MediaPipe index |
|---------------|-----------------|
| `eye_l_outer` | 33  |
| `eye_l_inner` | 133 |
| `eye_r_inner` | 362 |
| `eye_r_outer` | 263 |
| `iris_l_center` | 468 (requires `refine_landmarks=True`) |
| `iris_r_center` | 473 |
| `nose_tip` | 1 |
| `nose_bottom` | 2 |
| `nose_wing_l` | 98 |
| `nose_wing_r` | 327 |
| `mouth_corner_l` | 61 |
| `mouth_corner_r` | 291 |
| `mouth_upper` | 13 |
| `mouth_lower` | 14 |
| `chin` | 152 |
| `forehead_top` | 10 (note: sits below the actual hairline — systematic underestimate of `hairline_to_brow_per_eye`) |
| `brow_center` | 9 |
| `brow_l_peak` | 105 |
| `brow_r_peak` | 334 |

## Ratio → MakeHuman modifier path

`MODIFIER_MAP` in `face_proportion_analyzer.py`. Convention:
`<group>/<target>-<min_keyword>|<max_keyword>`. Step C resolves to
`makehuman/data/targets/<group>/<target>-<min_keyword>.target` for
negative values and `<target>-<max_keyword>.target` for positive.

| Ratio key | MH modifier path | When subject_ratio < baseline | When subject_ratio > baseline |
|-----------|------------------|-------------------------------|-------------------------------|
| `face_width_per_eye` | `head/head-scale-horiz-decr|incr` | `head-scale-horiz-decr.target` (narrower) | `head-scale-horiz-incr.target` (wider) |
| `face_height_per_eye` | `head/head-scale-vert-decr|incr` | shorter face | taller face |
| `nose_width_per_eye` | `nose/nose-scale-horiz-decr|incr` | narrower nose | wider nose |
| `mouth_width_per_eye` | `mouth/mouth-scale-horiz-decr|incr` | narrower mouth | wider mouth |
| `hairline_to_brow_per_eye` | `forehead/forehead-scale-vert-decr|incr` | shorter forehead | taller forehead |
| `brow_to_nose_per_eye` | `nose/nose-scale-vert-decr|incr` | shorter nose vertical | taller nose vertical |
| `eye_to_mouth_per_eye` | `mouth/mouth-trans-down|up` | mouth moves up | mouth moves down |
| `intereye_per_eye` | `eyes/l-eye-trans-in|out` | left eye moves in | left eye moves out |
| `ipd_per_eye` | `eyes/r-eye-trans-in|out` | right eye moves in | right eye moves out |

## Sensitivity & clamping

In `face_proportion_analyzer.py`:
```python
SENSITIVITY = 1.5
MH_VALUE_CLAMP = (-1.0, 1.0)
```

The normalisation: `value = clip(delta / baseline × SENSITIVITY, −1.0, 1.0)`.

`SENSITIVITY = 1.5` was tuned so that a "typical" subject reaches modifier
values around ±0.3–0.5 and an extreme subject saturates around ±0.8–1.0.
Lower it (closer to 1.0) for more conservative morphs, raise (toward 2.0)
for more aggressive ones. Beyond 2.0 the MH targets begin distorting
geometry — eyes pop off the skull, nose stretches unnaturally.

`amplify` in `step_c_apply_modifiers.py` is applied **after** the clamp
above, so to push past ±1.0 you can pass `--amplify 2.0` (combines to
±2.0 final modifier weight) but most `.target` files are calibrated to
look reasonable in the ±1.0 range only.

## Ratios that don't drive an MH modifier

`MODIFIER_MAP` deliberately omits some ratios. The renderer and JSON still
expose them, they just don't morph the MH mesh. Notably:

- `nose_to_chin_per_eye` — there's no clean single-axis MH modifier for
  "lengthen chin" without affecting the rest of the head. Could be wired
  to `head/head-trans-down|up` or to a chin-specific modifier (see
  `extending.md`).
- `ipd_per_eye` is redundant with `intereye_per_eye` for the purposes of
  the eye-spacing modifier — both currently drive
  `eyes/l-eye-trans-in|out` and `eyes/r-eye-trans-in|out` respectively.
  Use both for symmetric eye-shift; drop one if you want one eye fixed.

## Verifying the mapping

After `face_proportion_analyzer.py` runs, look at
`outputs/data/<subject>_face_proportions.json`. The `modifier_targets`
field lists every modifier the analyzer proposes, with subject ratio,
baseline ratio, delta, and proposed value. This is the canonical
audit-trail for what gets applied in step C.

`step_c_apply_modifiers.py` then logs each `.target` file it actually
loaded, with its applied weight and the number of vertices displaced —
that's the runtime confirmation. Skipped or unresolved modifiers are
logged as warnings.
