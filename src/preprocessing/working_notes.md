# Preprocessing — Working Notes

## iNaturalist image filename construction

iNat images on GCS are stored as `{photo_uuid}.{extension}` — the extension is
*not* fixed (jpg / jpeg / png all appear). The `inat_image_quality` table is
keyed on `photo_uuid` only. To open a scored image (or to score a new one)
you must join `inat_image_quality` → `inat_filtered_observations` on
`photo_uuid` to retrieve the per-row `extension` and reconstruct the full
object name.

```sql
SELECT q.*, o.extension
FROM inat_image_quality q
JOIN inat_filtered_observations o USING (photo_uuid);
```

In Python, the GCS object key is `f"{photo_uuid}.{extension}"` under whichever
prefix the photo was transferred to (`training/`, `validation/`, etc.).

The LILA side does not have this complication — `lila_collected_images.file_name`
is the full filename including extension and is the FK target directly.

## iNat capture context: underwater vs above-water

iNat is a citizen-science platform; fish observations include underwater dive
photos, fishing-deck photos, aquarium shots, market photos, and lab specimens.
For a dive-companion classifier we want to bias training toward underwater
captures only. UIQM does **not** solve this — it actively rewards good
lighting and rich color, which means above-water shots score higher and
would dominate a "top-N by UIQM per class" filter.

Decision (2026-04-24): start with a **simple color-statistics heuristic**, not
CLIP, for v1. Reasoning:

- Pure NumPy, ~1ms/image, runs CPU-side; ~30–60 min for 1M iNat photos.
- Underwater attenuates red dramatically — `mean(R) / (mean(R)+mean(G)+mean(B))`
  is a strong univariate signal (~0.15–0.25 underwater vs ~0.30–0.45 above-water).
- We store the raw channel means in the table so thresholds are tunable later
  without re-scoring. Same pattern as UIQM sub-scores.
- If the heuristic underperforms on a manually-checked sample, CLIP-ViT-B/32 is
  a 2–4 hour GPU upgrade with the same table schema (just add a `clip_score`
  column in a follow-up migration).

Capture context only applies to iNat — LILA's 17 sources are all aquatic
capture contexts, so no LILA equivalent is needed.

Filtering pipeline order matters: **capture-context filter → UIQM filter →
top-N per class**. Apply the underwater gate first so UIQM is computed only
within the relevant subset and "best-quality underwater photo" is what wins.

## Per-class downsampling target

At 1500 species × 500 images/class = ~750K training images. Selection rule
within each class:

1. Filter to `is_underwater = TRUE` (heuristic).
2. Rank remaining images by UIQM (composite or weighted sub-score TBD).
3. Take top 500 (or all if fewer remain — log per-class actual count).

This means the runner that scores iNat needs both UIQM and capture-context
columns populated before per-class selection happens.
