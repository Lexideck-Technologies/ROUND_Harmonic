
# version 0.2.0
import subprocess
import os
import sys
import uuid

scripts = [
    "benchmark_brackets.py",
    "benchmark_clock.py",
    "benchmark_parity.py",
    "benchmark_topology.py"
]

print("Starting Full Test Battery...")
BATCH_UID = str(uuid.uuid4())[:8]
os.environ["ROUND_BATCH_UID"] = BATCH_UID
print(f"Batch UUID: {BATCH_UID}")
print(f"Current Directory: {os.getcwd()}")


for script in scripts:
    print(f"\n=========================================")
    print(f"Running {script}...")
    print(f"=========================================\n")
    try:
        # verify file exists
        if not os.path.exists(script):
            print(f"Script {script} not found!")
            continue
            
        subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

print("\nBattery Complete.")
