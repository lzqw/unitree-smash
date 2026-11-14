set pagination off
set confirm off
set debuginfod enabled off
handle SIGPIPE nostop noprint pass
run
printf "\n=== STOP REASON ===\n"
p $_siginfo
printf "\n=== CURRENT THREAD ===\n"
bt 80
printf "\n=== ALL THREADS ===\n"
thread apply all bt 6
printf "\n=== LOADED LIBRARIES ===\n"
info sharedlibrary
