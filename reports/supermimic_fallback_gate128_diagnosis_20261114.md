# Fallback gate diagnosis

The fallback gate at eight terminated episodes was still too early. In the sixth run, the first actor update occurred with only eight failed episodes, zero qualified airborne string contacts, and imagined return around `-11`; critic loss was around `36`. The actor update was stopped and the run was preserved.

The default fallback threshold is now `128` terminated episodes. This forces broader native coverage before imagined updates when no full timeout is available. The preferred full-timeout gate remains unchanged. The next run will keep actor learning rate `1e-5`, anchor coefficient `0.2`, teacher-forced world fitting, and the same task/timebase.

