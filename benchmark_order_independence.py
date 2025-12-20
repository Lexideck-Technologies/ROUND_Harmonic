# version 0.7.3 - "The Hyper-Resolution Basin" (Order Independence Gauntlet)
import torch
import numpy as np
import os
import uuid
import random
from benchmark_long_term import run_long_term_comparison, WORDS, LONG_TERM_CONFIG

# --- Configuration ---
RUNS = 3 # Deep Gauntlet: 3 full dual-model batches
UID = os.environ.get('ROUND_BATCH_UID', str(uuid.uuid4())[:8])
OUTPUT_DIR = os.environ.get('ROUND_OUTPUT_DIR', 'data')

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    log_path = os.path.join(OUTPUT_DIR, f'log_order_independence_{UID}.txt')
    L_FILE = open(log_path, 'w')
    def P(s): print(s); L_FILE.write(str(s) + '\n'); L_FILE.flush()

    P(f"--- [ORDER INDEPENDENCE GAUNTLET v0.6.2] ---")
    P(f"Starting Head-to-Head Brutality Test: {RUNS} Shuffled Batches")
    
    master_results = []
    
    for i in range(1, RUNS + 1):
        P(f"\n>>> INDEPENDENCE RUN {i}/{RUNS} Starting...")
        shuffled = list(WORDS)
        random.shuffle(shuffled)
        
        # Call the modular long-term engine
        run_res = run_long_term_comparison(
            shuffled_words=shuffled,
            epochs=LONG_TERM_CONFIG['EPOCHS'],
            hidden_size_r=LONG_TERM_CONFIG['HIDDEN_R'],
            hidden_size_g=LONG_TERM_CONFIG['HIDDEN_G'],
            p_func=P,
            output_dir=OUTPUT_DIR,
            plot_name=f"benchmark_order_independence_{UID}_{i}.png"
        )
        master_results.append(run_res)
        
        P(f"<<< RUN {i} COMPLETE.")

    # Final Comparative Summary
    P("\n" + "="*50)
    P("FINAL GAUNTLET SUMMARY (Aggregate Shuffled Order)")
    P("="*50)
    header = f"{'Word':12s} | {'ROUND Mean':13s} | {'GRU Mean':13s} | {'Status'}"
    P(header); P("-" * len(header))
    
    for word in WORDS:
        r_accs = [res[word][0] for res in master_results]
        g_accs = [res[word][1] for res in master_results]
        r_mean = np.mean(r_accs) * 100
        g_mean = np.mean(g_accs) * 100
        
        status = "PERFECT" if r_mean == 100 else "STABLE"
        diff = r_mean - g_mean
        P(f"{word:12s} | {r_mean:10.1f}% | {g_mean:10.1f}% | {status} (+{diff:.1f}%)")
    
    P("="*50)
    P(f"Gauntlet Log: {log_path}")
    L_FILE.close()

if __name__ == "__main__":
    main()
