#!/usr/bin/env python3
"""
Generate randomized level sequences with constraints and per-variant minimum hits.

Output format:
const sequences = [
  ['old_s111_1', 'old_s112_1', ...],
  ...
];

Key constraints:
- Each sequence starts with old_s111_1 and old_s112_1
- Each base level has two variants: _1 and _2
- _1 and _2 variants of the same base cannot appear in the same sequence
- Exclusive pairs and set A/B distance rules still apply
"""

import random
from typing import Dict, List, Optional
import math

BASES = [
    "sm211", "sm221", "sm431", "sm611", "sm543"
]

SET_A = {"s211", "s311", "s411", "s511"}
SET_B = {"s221", "s321", "s421", "s521"}
EXCLUSIVE_PAIRS = {
    ("s531", "s532"),
    ("s431", "s432"),
    ("s441", "s442"),
    ("s351", "s352"),
    ("s531", "s532"),
    ("s541", "s542"),
    ("s544", "s543"),
}

# Configurable parameters
NUM_SEQUENCES = 10
SEQ_LENGTH = 10 # Will be 10 after prepending old_s111_1 and old_s112_1
MIN_HITS_PER_VARIANT = 1  # each variant (old_sXXX_1 or old_sXXX_2) must appear at least this many times across all sequences

MAX_GLOBAL_TRIES = 2000
MAX_LOCAL_TRIES = 20000  # backtracking attempts per sequence fill

# Build variant keys including old_ prefix, both _1 and _2 variants
VARIANTS = [f"old_{b}_1" for b in BASES] + [f"old_{b}_2" for b in BASES]

def base_of(variant_key: str) -> str:
    # variant_key like "old_s111_1" -> "s111"
    parts = variant_key.split("_")
    return parts[1] if len(parts) >= 2 else variant_key

def variant_number(variant_key: str) -> str:
    # variant_key like "old_s111_1" -> "1"
    parts = variant_key.split("_")
    return parts[2] if len(parts) >= 3 else "1"

def violates_exclusive(candidate_base: str, current_bases: List[str]) -> bool:
    for a, b in EXCLUSIVE_PAIRS:
        if candidate_base == a and b in current_bases:
            return True
        if candidate_base == b and a in current_bases:
            return True
    return False

def compatible_variant(candidate_key: str, pos: int, current_keys: List[str]) -> bool:
    cand_base = base_of(candidate_key)
    cand_variant = variant_number(candidate_key)
    cur_bases = [base_of(k) for k in current_keys]

    # NEW: Check if _1 and _2 variants of the same base already exist in this sequence
    for k in current_keys:
        if base_of(k) == cand_base and variant_number(k) != cand_variant:
            return False

    # exclusive-pair rule (bases)
    if violates_exclusive(cand_base, cur_bases):
        return False

    # set A rule: max 2 per sequence, distance >= 4 between occurrences (i.e., at least 3 in between)
    if cand_base in SET_A:
        existing_positions = [i for i, k in enumerate(current_keys) if base_of(k) in SET_A]
        if len(existing_positions) >= 2:
            return False
        for p in existing_positions:
            if abs(pos - p) < 4:
                return False

    # set B rule similar
    if cand_base in SET_B:
        existing_positions = [i for i, k in enumerate(current_keys) if base_of(k) in SET_B]
        if len(existing_positions) >= 2:
            return False
        for p in existing_positions:
            if abs(pos - p) < 4:
                return False

    return True

def fill_sequence_from_pool(seq_len: int, pool: Dict[str,int], original_pool: Dict[str,int]) -> Optional[List[str]]:
    """
    Attempt to build one sequence of keys (variant_keys like 'old_s111_1') from pool (counts).
    Uses backtracking. Caller must decrement pool on success.
    """
    attempts = 0

    def helper(current: List[str], local_pool: Dict[str,int]) -> Optional[List[str]]:
        nonlocal attempts
        if attempts > MAX_LOCAL_TRIES:
            return None
        attempts += 1
        if len(current) == seq_len:
            return current
        # candidates available
        candidates = [k for k,cnt in local_pool.items() if cnt > 0]
        # Prefer candidates that still need many placements (to balance) and that have been used less so far
        def sort_key(k):
            remaining = local_pool[k]
            assigned = original_pool.get(k, 0) - local_pool.get(k, 0)
            # higher remaining -> earlier, lower assigned -> earlier
            return (-remaining, assigned, random.random())
        candidates.sort(key=sort_key)

        pos = len(current)
        for cand in candidates:
            if compatible_variant(cand, pos, current):
                local_pool[cand] -= 1
                current.append(cand)
                res = helper(current, local_pool)
                if res is not None:
                    return res
                # backtrack
                current.pop()
                local_pool[cand] += 1
        return None

    pool_copy = dict(pool)
    return helper([], pool_copy)

def make_sequences(num_sequences: int, seq_length: int, min_hits: int) -> List[List[str]]:
    total_slots = num_sequences * seq_length
    num_variants = len(VARIANTS)
    required_slots = num_variants * min_hits
    if required_slots > total_slots:
        raise ValueError(f"Not enough total slots ({total_slots}) to meet min_hits {min_hits} for {num_variants} variants (need {required_slots}).")

    # build initial pool: each variant repeated min_hits times
    base_pool: Dict[str,int] = {v: min_hits for v in VARIANTS}
    remaining_slots = total_slots - required_slots

    for global_try in range(MAX_GLOBAL_TRIES):
        pool = dict(base_pool)
        # EVENLY distribute remaining slots across variants first (balanced)
        if remaining_slots > 0:
            per_variant = remaining_slots // num_variants
            rem = remaining_slots % num_variants
            if per_variant:
                for v in VARIANTS:
                    pool[v] = pool.get(v, 0) + per_variant
            if rem:
                # give the leftover 1-by-1 to a random shuffle of variants
                vs = VARIANTS[:]
                random.shuffle(vs)
                for v in vs[:rem]:
                    pool[v] = pool.get(v, 0) + 1
        sequences: List[List[str]] = []
        success = True
        # copy of original pool to use for balancing during fills
        original_pool = dict(pool)
        # Fill each sequence in order
        for si in range(num_sequences):
            seq = fill_sequence_from_pool(seq_length, pool, original_pool)
            if seq is None:
                success = False
                break
            # commit removals from pool (decrement counts)
            for k in seq:
                pool[k] -= 1
                if pool[k] < 0:
                    success = False
            if not success:
                break
            sequences.append(seq)

        if success:
            return sequences
    raise RuntimeError("Unable to construct sequences with given constraints after many tries")

def format_sequences_js(seqs: List[List[str]]) -> (str, Dict[str,int]):
    # Output each sequence item as the variant key itself (e.g., "old_s111_1")
    usage: Dict[str,int] = {v:0 for v in VARIANTS}
    # Add old_s111_1 and old_s112_1 to usage tracking
    usage["old_s111_1"] = 0
    usage["old_s112_1"] = 0

    formatted_seqs: List[List[str]] = []
    for seq in seqs:
        out_seq: List[str] = ["old_s111_1", "old_s112_1"]  # Prepend the required first two items
        usage["old_s111_1"] += 1
        usage["old_s112_1"] += 1

        for v in seq:
            usage[v] += 1
            out_seq.append(f"{v}")
        formatted_seqs.append(out_seq)
    # render JS
    lines = ["const sequences = ["]
    for s in formatted_seqs:
        items = ", ".join(f"'{item}'" for item in s)
        lines.append(f"  [{items}],")
    lines.append("];")
    return "\n".join(lines), usage

def logistic_tracker(usage: Dict[str,int], min_hits: int) -> str:
    # Summarize usage per variant and per base
    total_used = sum(usage.values())
    lines = []
    lines.append(f"Total assignments: {total_used}")
    # per-variant
    below = []
    for v in sorted(usage.keys()):
        cnt = usage[v]
        lines.append(f"{v}: {cnt}")
        if cnt < min_hits:
            below.append((v, cnt))
    # per-base totals
    base_totals: Dict[str,int] = {}
    for v, cnt in usage.items():
        b = base_of(v)
        base_totals[b] = base_totals.get(b, 0) + cnt
    lines.append("")
    lines.append("Per-base totals:")
    for b in sorted(base_totals.keys()):
        lines.append(f"{b}: {base_totals[b]}")
    if below:
        lines.append("")
        lines.append(f"Variants below min_hits ({min_hits}):")
        for v,c in below:
            lines.append(f"{v}: {c}")
    return "\n".join(lines)

if __name__ == "__main__":
    random.seed()  # set an int for reproducible runs, e.g. random.seed(123)
    seqs = make_sequences(NUM_SEQUENCES, SEQ_LENGTH, MIN_HITS_PER_VARIANT)
    js, usage = format_sequences_js(seqs)
    print(js)
    print("\n--- Logistic tracker ---")
    print(logistic_tracker(usage, MIN_HITS_PER_VARIANT))