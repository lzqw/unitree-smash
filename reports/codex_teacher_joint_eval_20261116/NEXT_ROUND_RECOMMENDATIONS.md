# Next round recommendations

This round is stopped after one native evaluation. Do not start another candidate automatically.

1. **Recovery direct-search state handoff**
   - Evidence: 333 entry attempts, 0 adopted sequences; all failed with `ValueError: Actual command history required`.
   - Failure mechanism: the repair worker does not receive the native command-history state required by the predictor.
   - Change: construct the direct-search worker from the current native command-history snapshot, including previous final target and all predictor history fields; keep the 32-node full physical validation and first-command freedom.
   - Verification: unit test that a worker created at an actual Recovery handoff can execute `_direct_search_after_both_fail` without the history error, then a CPU bounded smoke; only afterward consider one new native candidate.

2. **Recovery adoption accounting**
   - Evidence: native `pure_control_result.json` reports `sequence_mpc=false` and direct search adopted 0; this metadata is inconsistent with the intended candidate contract.
   - Change: add explicit counters for direct-search calls, feasible candidates, adopted sequences, and fallback reasons in the native result writer.
   - Verification: compare counters against `pure_control_diagnostics.jsonl` and retain the 2e-6 command replay check.

3. **Approach/WBC**
   - Evidence: this run had 29 / 128 airborne contacts and 21 / 128 Q; no separate Approach/WBC ablation was run.
   - Change: keep Approach/WBC frozen until the Recovery handoff bug is fixed; do not attribute this run to a WBC gain.
   - Verification: use the same 128 protocol and explicit stage ownership audit in a later approved candidate.

4. **Lifecycle**
   - Evidence: teacher service completed normally in its dedicated cgroup; external SuperMimic failed with its own `KeyError` and was not restarted.
   - Change: keep teacher service wrapper/observer. SuperMimic requires a separately authorized persistent launch after its owner fixes its own replay schema.
   - Verification: no cross-line signals, separate PGID/SID/cgroup, and independent exit records.
