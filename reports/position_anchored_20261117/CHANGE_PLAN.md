# Implemented change / acceptance map

1. Preserve position first: implemented and same-input G0 passed.
2. Correct normal without losing position: implemented, tests passed; real fixed inputs still fail simultaneous pose gate.
3. Bound contact speed and recompute actual return: implemented in production branch, synthetic path tests passed; fixed inputs stop before speed solve.
4. Fail closed physics and identify RR cause: implemented; optional mechanically coupled stops remove early violation, but full paired replay still fails later contact convergence.
5. Replace untrusted long forecast and complete online lifecycle: not completed. Current forecast remains offline reference; native anchored path disabled. No asynchronous worker/feedback closure claimed.
6. Runtime profiling: implemented partial CPU timings, p95 fails100ms. No new native run allowed.
7. Deliver real code, tests and preserved failures: REPORT.md, COMMANDS.md, CPU_EXIT.json, GATES.json, compressed raw replays and SOURCE_DIFF.patch.gz.

This is an engineering checkpoint, not a full accepted candidate. Do not automatically advance to a native probe or128 panel.
