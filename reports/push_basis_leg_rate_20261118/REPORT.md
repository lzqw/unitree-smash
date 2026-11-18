# Push-window leg-rate attempt: NOT_COMPLETED

This attempt tested a diagnostic leg-rate basis normalized to the measured
support window. It was not completed: the process ended with exit 143 after
33 derivative rollouts, before a proposal was fully evaluated. No result was
selected, no command was installed, and no performance claim is made.

The unverified basis change was reverted. The default and previously validated
leg-rate basis remain unchanged. Existing native assets, physics, scorer,
Recovery and SuperMimic were untouched. Preserve `replay.log` and `replay.exit`
as the failure evidence.

`evaluation_stage=teacher_debug`, `formal_eligible=false`,
`formal_acceptance=false`, `formal_cases_read=false`, `NATIVE_HIT=NOT_RUN`.
