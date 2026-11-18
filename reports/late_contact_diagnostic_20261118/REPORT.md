# Single late-contact diagnostic

A single causal hold-tail diagnostic extended the selected 0.33 s prefix by two
unchanged command/reference ticks to 0.35 s. It did not optimize, install a
reference, read future native states, or alter the original physical model.

The model path remained valid, but terminal contact position error was
**83.1302 mm**, worse than the selected 0.33 s result (**67.2137 mm**). This
rejects late timing as the immediate remedy. The 0.35 s point is diagnostic only
and is not added to the frozen event window or any candidate set.

`NATIVE_HIT=NOT_RUN`, `PHYSX_REFERENCE_VALID=NOT_RUN`,
`evaluation_stage=teacher_debug`, `formal_eligible=false`,
`formal_acceptance=false`, `formal_cases_read=false`.
