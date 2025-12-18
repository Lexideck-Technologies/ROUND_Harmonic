
# version 0.6.2 - Harmonic Monism
import subprocess
import os
import sys
import uuid

scripts = [
    "benchmark_brackets_masked.py",
    "benchmark_parity.py",
    "benchmark_topology.py",
    "benchmark_ascii.py",
    "benchmark_colors.py",
    "benchmark_oracle.py",
    "benchmark_permutations.py",
    "benchmark_long_term.py",
    "benchmark_order_independence.py"
]

print("Starting Full Test Battery...")
BATCH_UID = str(uuid.uuid4())[:8]
os.environ["ROUND_BATCH_UID"] = BATCH_UID
OUTPUT_DIR = os.path.join("data", BATCH_UID)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
os.environ["ROUND_OUTPUT_DIR"] = OUTPUT_DIR

print(f"Batch UUID: {BATCH_UID}")
print(f"Output Directory: {OUTPUT_DIR}")
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
