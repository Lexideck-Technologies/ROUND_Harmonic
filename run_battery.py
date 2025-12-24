# version 0.8.0 - "The Frozen Basin"
import subprocess
import os
import sys
import uuid

scripts = [
    "benchmarks/benchmark_brackets_masked.py",
    "benchmarks/benchmark_parity.py",
    "benchmarks/benchmark_topology.py",
    "benchmarks/benchmark_ascii.py",
    "benchmarks/benchmark_colors.py",
    "benchmarks/benchmark_oracle.py",
    "benchmarks/benchmark_permutations.py",
    "benchmarks/benchmark_phase_lock.py",
    "benchmarks/benchmark_order_independence.py",
    "benchmarks/benchmark_mod17.py"
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
