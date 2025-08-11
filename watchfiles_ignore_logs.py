# Watchfiles configuration to ignore log files and prevent infinite change detection loops
from watchfiles import watch
import os
import sys

watch_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend'))

for changes in watch(watch_path):
    filtered = [c for c in changes if not (c[1].endswith('.log') or c[1].endswith('.pyc'))]
    if filtered:
        print(filtered)

