import glob
import json
import os

folders = ["HELLO", "THANK_YOU", "YES"]

for lab in folders:
    files = glob.glob(os.path.join("data", "static", lab, "*.json"))
    print(lab, "files:", len(files))
    if not files:
        continue
    with open(files[0], "r") as f:
        vec = json.load(f)
    print("  first sample length:", len(vec))
