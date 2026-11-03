# Perio charting output (Harmony Dental AI -> Harmony charting tool)

The model MEASURES; Harmony's charting code applies the rules (staging, grading,
merging with probe/voice data). Output for one radiograph:

```json
{
  "type": "xray", "film": "periapical|bitewing|panoramic|occlusal",
  "teeth": [
    {
      "fdi": "36", "present": true, "confidence": 0.94,
      "box": [x1, y1, x2, y2],
      "restorations": [{"kind": "crown|filling|implant|root canal treatment", "confidence": 0.88}],
      "bone_level": {
        "mesial": {"loss_pct": 35.2, "cej": [x, y], "crest": [x, y], "apex": [x, y], "confidence": 0.71},
        "distal": {"loss_pct": 18.0, "...": "..."}
      },
      "findings": [{"kind": "caries|deep caries|periapical lesion|badly decayed", "confidence": 0.62, "box": [x1, y1, x2, y2]}]
    }
  ],
  "missing_teeth": ["18", "28"],
  "model_version": "harmony-dental-vN"
}
```

- `loss_pct` = distance(CEJ, bone crest) / distance(CEJ, apex) x 100, minus the normal
  1.5-2 mm crest gap only when pixel spacing is known. Percent of root length is what the
  2017 AAP/EFP staging uses (Stage I <15%, II 15-33%, III/IV >33%), so millimetres are optional.
- A site is omitted (not guessed) when its CEJ, crest or apex is not visible.
- Pocket depth, bleeding, mobility and furcation are NOT read from X-rays; they come from
  the probe / voice modes. Furcation needs owner-labelled data (no commercial-safe public set).
- Confidence is the detector score (0-1); Harmony should show anything < 0.5 as "check".
- How each charting mode uses it: template/voice -> pre-fill present/missing teeth and
  restorations, cross-check dictated values; X-ray mode -> fill bone level per site and
  suggest a stage for the dentist to confirm.
