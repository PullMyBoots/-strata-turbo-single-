# Native batch media experiments (not selected)

Placement restoration, one-token media verification, GPU-only missing experts, and batched media prompt tails were investigated after Swift returned only timestamps for tiny videos. Some initial smoke tests passed, but each numerical-engine approach failed a later regression. These candidates are not the selected fix and are excluded from the final comparison.

The successful controlled intervention raises the minimum visual token budget for Swift inputs. The unchanged Strata numerical engine passed the same checks with that input setting, including retrieval near the native context limit and mixed six-request checks. See the final video-resolution report and complete matrix for validation. This does not establish broad model-quality equivalence.

`build.py` and `source.patch` retain the final rejected prototype for reproducibility. Build after the direct-slot-state candidate:

```bash
.venv/bin/python experimental/native-media-profile/build.py
```

The opt-in flag is `STRATA_MEDIA_FIXED_PROFILE=1`. Build artifacts retain command lines and hashes. Do not select this prototype for deployment.
