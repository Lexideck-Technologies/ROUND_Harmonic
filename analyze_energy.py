def str_to_bits(s):
    return [[int(c) for c in format(ord(ch), '08b')] for ch in s]

WORDS = ["COGITATING", "TOPOLOGY", "SYMMETRY", "MONISM", "RESONANCE", "SINGULARITY"]

for w in WORDS:
    bits = str_to_bits(w)
    total = sum([sum(b) for b in bits])
    print(f"{w} -> {total} total bits, avg {total/len(w):.2f}")
