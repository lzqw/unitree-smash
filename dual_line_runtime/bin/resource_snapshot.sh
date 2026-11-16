#!/usr/bin/env bash
set -euo pipefail
out=${1:?output directory}
mkdir -p "$out"
date --iso-8601=seconds > "$out/time.txt"
ps -eo user,pid,ppid,pgid,sid,lstart,etime,stat,%cpu,%mem,rss,args --sort=pid > "$out/processes.txt"
free -h > "$out/memory.txt"
cat /proc/pressure/memory > "$out/pressure_memory.txt"
nproc > "$out/cpu_count.txt"
nvidia-smi -q > "$out/nvidia_smi.txt" 2>&1 || true
systemctl --user list-units --all --no-pager > "$out/user_units.txt" 2>&1 || true
