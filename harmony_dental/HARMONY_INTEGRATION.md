# Plugging Harmony Dental AI into Harmony

1. Run the server (GPU optional): `HARMONY_MODEL=harmony-dental.pt uvicorn serve:app --port 8080`
   from `harmony_dental/` (deps: torch, monai, SimpleITK, fastapi, uvicorn, python-multipart).
2. Harmony (Supabase edge function or backend) sends each uploaded record:
   `POST /analyze` with `file` and optional `film`. Patient X-rays stay in Harmony's Supabase
   storage; the server keeps nothing.
3. Response for X-rays = perio charting JSON (see `perio/SPEC.md`): teeth (FDI, present,
   restorations, findings, bone_level mesial/distal %, suggested_stage, confidence) plus
   `missing_teeth` and `all_findings`. CBCT returns structures with volumes.
4. Charting tool use:
   - Template / voice modes: pre-fill present/missing teeth and restorations; flag
     disagreements between dictation and the X-ray.
   - X-ray mode: fill bone level per site, show suggested stage; anything with
     confidence < 0.5 is shown as "check". The dentist confirms every value.
5. Versioning: `/health` returns the model version and its held-out scores; keep the previous
   model file for instant rollback.
