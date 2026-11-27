## R13 Recovery candidates on native64

All 20 BPTT updates completed in 636 seconds. Both candidates preserved 54 qualified contacts and had zero robot net contacts. Each candidate was paired against the incumbent in its own 64-lane slot.

| Candidate | Body ground among qualified | Strict | Relaxed | Ball success |
|---|---:|---:|---:|---:|
| Conservative | 31 → 30 | 3 → 5 | 9 → 11 | 46 → 45 |
| Moderate | 32 → 31 | 3 → 6 | 8 → 14 | 46 → 45 |

Both lost A-991101-002 with final_best. For the conservative candidate, landing x changed from 3.9809 to 4.0350 m; for moderate it changed from 3.9198 to 4.0350 m. All four executions recorded the same first-contact and release timestamps and legal net crossing. The matched incumbent slots already differed in outgoing velocity and landing coordinate on this case. Thus the panel loss is real under the configured guard; the observations do not isolate whether the policy caused the release difference. No candidate promoted. A diagnostic fixed128 run for the conservative candidate started.
