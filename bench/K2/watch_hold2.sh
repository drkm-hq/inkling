#!/bin/zsh
# log watch whose process name does not match the competing-process regexes
cd ~/code/inkling-contrib/bench/results
tail -n 0 -F K2-hold2.log K2-bench-mlxvlm-same-hold.log K2-bench-mlxlm-2bit-same-hold.log K2-hold3.log 2>/dev/null | grep -E --line-buffered "rc=|Traceback|Error|FAILED|Killed|hold |load:|run |SUMMARY|warm-up|released|no time"
