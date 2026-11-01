#!/bin/zsh
# K3: before/after prompt-cache measurement on Inkling-Small under one GPU_LOCK hold (bench/baseline/with_gpu_lock.sh
# waits for the lock and for big processes, writes its PID into bench/GPU_LOCK, removes it when run_four.sh exits).
exec $HOME/code/inkling-contrib/bench/baseline/with_gpu_lock.sh $HOME/code/inkling-contrib/package/k3-llamacpp-prompt-cache/bench/run_four.sh
