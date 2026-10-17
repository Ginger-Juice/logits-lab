#!/bin/bash
# kill only python processes whose argv[1] is scripts/gen.py
for p in $(ps -eo pid=,args= | awk '$2 ~ /python$/ && $3 == "scripts/gen.py" {print $1}'); do echo kill $p; kill $p; done
