#!/usr/bin/env python3
"""
Generate randomized level sequences for exp4 levels with constraints.

Output format:
const sequences = [
  ['sm111_1', 'sm112_1', 'sm211_1', 'sm321_1', ...],
  ...
];

Key constraints:
- Each sequence starts with sm111_1 and sm112_1 (tutorial levels)
- 10 main experiment levels per sequence (12 total with tutorials)
- Same-base variants (_1 and _2) cannot appear in the same sequence
- Set A rule: max 2 from {sm211, sm311, sm411, sm511} per sequence, distance >= 4
- Set B rule: max 2 from {sm221, sm321, sm421, sm521} per sequence, distance >= 4
- Exclusive pairs cannot appear in the same sequence
"""

import random
from typing import Dict, List, Optional

# Base levels from levels.txt
BASE_LEVELS = [
    "sm211", "sm221", "sm311", "sm321", "sm331", "sm341", "sm351", "sm361", "sm371",
    "sm411", "sm421", "sm431", "sm432", "sm511", "sm521", "sm531", "sm541", "sm543",
    "sm611", "sm612"
]

# Available exp4 level variants (each base has _1 and _2)
VARIANTS = []
for base in BASE_LEVELS:
    VARIANTS.append(f"{base}_1")
    VARIANTS.append(f"{base}_2")

# Set A and Set B rules (max 2 per sequence, distance >= 4)
SET_A = {"sm211", "sm311", "sm411", "sm511"}
SET_B = {"sm221", "sm321", "sm421", "sm521"}

# Exclusive pairs: these bases cannot appear in the same sequence
EXCLUSIVE_PAIRS = [
    ("sm611", "sm612"),
    ("sm431", "sm432"),
]

# Configurable parameters
NUM_SEQUENCES = 10
SEQ_LENGTH = 10  # Will be 12 after prepending sm111_1 and sm112_1
MIN_HITS_PER_VARIANT = 1  # each variant must appear at least this many times

MAX_GLOBAL_TRIES = 2000
MAX_LOCAL_TRIES = 50000  # backtracking attempts per sequence fill

def base_of(variant_key: str) -> str:
    """Extract base level ID from variant key.

    Examples:
        'sm211_1' -> 'sm211'
        'sm351_3' -> 'sm351'
    """
    parts = variant_key.split("_")
    return parts[0] if len(parts) >= 1 else variant_key


def violates_exclusive(candidate_base: str, current_bases: List[str]) -> bool:
    """Check if candidate violates exclusive pair rules."""
    for a, b in EXCLUSIVE_PAIRS:
        if candidate_base == a and b in current_bases:
            return True
        if candidate_base == b and a in current_bases:
            return True
    return False


def compatible_variant(candidate_key: str, pos: int, current_keys: List[str]) -> bool:
    """Check if candidate variant is compatible with current sequence.

    Constraints:
    1. Same-base variants (_1 and _2) cannot appear in the same sequence
    2. Exclusive pairs cannot appear in the same sequence
    3. Set A: max 2 per sequence, distance >= 4 between occurrences
    4. Set B: max 2 per sequence, distance >= 4 between occurrences

    Args:
        candidate_key: Variant to check (e.g., 'sm211_1')
        pos: Position in sequence where candidate would be placed
        current_keys: Variants already in the sequence

    Returns:
        True if candidate is compatible, False otherwise
    """
    cand_base = base_of(candidate_key)
    cur_bases = [base_of(k) for k in current_keys]

    # Check 1: same-base variants cannot be together
    if cand_base in cur_bases:
        return False

    # Check 2: exclusive pair rule
    if violates_exclusive(cand_base, cur_bases):
        return False

    # Check 3: Set A rule - max 2 per sequence, distance >= 4
    if cand_base in SET_A:
        existing_positions = [i for i, k in enumerate(current_keys) if base_of(k) in SET_A]
        if len(existing_positions) >= 2:
            return False
        for p in existing_positions:
            if abs(pos - p) < 4:
                return False

    # Check 4: Set B rule - max 2 per sequence, distance >= 4
    if cand_base in SET_B:
        existing_positions = [i for i, k in enumerate(current_keys) if base_of(k) in SET_B]
        if len(existing_positions) >= 2:
            return False
        for p in existing_positions:
            if abs(pos - p) < 4:
                return False

    return True

def fill_sequence_from_pool(
    seq_len: int,
    pool: Dict[str, int],
    original_pool: Dict[str, int],
    sequence_num: int = 0
) -> Optional[List[str]]:
    """Attempt to build one sequence from pool using backtracking.

    Args:
        seq_len: Number of levels to generate
        pool: Current pool of available variants with counts
        original_pool: Original pool for balancing heuristics
        sequence_num: Sequence number for progress reporting

    Returns:
        List of variant keys if successful, None if failed
    """
    attempts = 0
    last_progress_print = 0

    def helper(current: List[str], local_pool: Dict[str, int]) -> Optional[List[str]]:
        nonlocal attempts, last_progress_print
        if attempts > MAX_LOCAL_TRIES:
            return None
        attempts += 1

        # Print progress every 10000 attempts
        if attempts - last_progress_print >= 10000:
            print(f"  Sequence {sequence_num}: {attempts} attempts, current length: {len(current)}/{seq_len}", flush=True)
            last_progress_print = attempts

        if len(current) == seq_len:
            return current

        # Get available candidates
        candidates = [k for k, cnt in local_pool.items() if cnt > 0]
        if not candidates:
            return None

        # Prefer candidates that still need many placements (for balance)
        def sort_key(k):
            remaining = local_pool[k]
            assigned = original_pool.get(k, 0) - local_pool.get(k, 0)
            # Higher remaining -> earlier, lower assigned -> earlier
            return (-remaining, assigned, random.random())
        candidates.sort(key=sort_key)

        # Randomize order more aggressively to avoid getting stuck
        if len(candidates) > 1:
            top_n = min(5, len(candidates))
            top_candidates = candidates[:top_n]
            rest = candidates[top_n:]
            random.shuffle(rest)
            candidates = top_candidates + rest

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
    """Generate all level sequences with balanced distribution.

    Args:
        num_sequences: Number of sequences to generate
        seq_length: Number of main levels per sequence
        min_hits: Minimum appearances per variant

    Returns:
        List of sequences, each containing variant keys

    Raises:
        ValueError: If constraints are impossible to satisfy
        RuntimeError: If generation fails after many attempts
    """
    total_slots = num_sequences * seq_length
    num_variants = len(VARIANTS)
    required_slots = num_variants * min_hits

    if required_slots > total_slots:
        raise ValueError(
            f"Not enough total slots ({total_slots}) to meet min_hits {min_hits} "
            f"for {num_variants} variants (need {required_slots})."
        )

    # Build initial pool: each variant repeated min_hits times
    base_pool: Dict[str, int] = {v: min_hits for v in VARIANTS}
    remaining_slots = total_slots - required_slots

    for global_try in range(MAX_GLOBAL_TRIES):
        pool = dict(base_pool)

        # Evenly distribute remaining slots across variants
        if remaining_slots > 0:
            per_variant = remaining_slots // num_variants
            rem = remaining_slots % num_variants
            if per_variant:
                for v in VARIANTS:
                    pool[v] = pool.get(v, 0) + per_variant
            if rem:
                # Give leftover slots to random variants
                vs = VARIANTS[:]
                random.shuffle(vs)
                for v in vs[:rem]:
                    pool[v] = pool.get(v, 0) + 1

        sequences: List[List[str]] = []
        success = True
        original_pool = dict(pool)

        # Fill each sequence in order
        for si in range(num_sequences):
            print(f"Attempting sequence {si + 1}/{num_sequences} (global try {global_try + 1}/{MAX_GLOBAL_TRIES})...", flush=True)
            seq = fill_sequence_from_pool(seq_length, pool, original_pool, sequence_num=si + 1)
            if seq is None:
                print(f"  Failed to fill sequence {si + 1}", flush=True)
                success = False
                break
            print(f"  Successfully filled sequence {si + 1}", flush=True)

            # Commit removals from pool
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

def format_sequences_js(seqs: List[List[str]]) -> tuple[str, Dict[str, int]]:
    """Format sequences as JavaScript code and track usage statistics.

    Args:
        seqs: List of sequences to format

    Returns:
        Tuple of (JavaScript code string, usage statistics dictionary)
    """
    usage: Dict[str, int] = {v: 0 for v in VARIANTS}
    # Add tutorial levels to usage tracking
    usage["sm111_1"] = 0
    usage["sm112_1"] = 0

    formatted_seqs: List[List[str]] = []
    for seq in seqs:
        # Prepend tutorial levels
        out_seq: List[str] = ["sm111_1", "sm112_1"]
        usage["sm111_1"] += 1
        usage["sm112_1"] += 1

        for v in seq:
            usage[v] += 1
            out_seq.append(v)
        formatted_seqs.append(out_seq)

    # Render JavaScript
    lines = ["const sequences = ["]
    for s in formatted_seqs:
        items = ", ".join(f"'{item}'" for item in s)
        lines.append(f"  [{items}],")
    lines.append("];")

    return "\n".join(lines), usage

def logistic_tracker(usage: Dict[str, int], min_hits: int) -> str:
    """Generate usage statistics report.

    Args:
        usage: Dictionary of variant usage counts
        min_hits: Target appearances per variant

    Returns:
        Formatted statistics string
    """
    total_used = sum(usage.values())
    lines = []
    lines.append(f"Total assignments: {total_used}")
    lines.append("")

    # Per-variant statistics
    below = []
    for v in sorted(usage.keys()):
        cnt = usage[v]
        lines.append(f"{v}: {cnt}")
        if cnt < min_hits and not v.startswith("sm111") and not v.startswith("sm112"):
            below.append((v, cnt))

    # Per-base totals
    base_totals: Dict[str, int] = {}
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
        for v, c in below:
            lines.append(f"{v}: {c}")

    return "\n".join(lines)

if __name__ == "__main__":
    random.seed()  # Use random seed; set an int for reproducible runs

    print("=" * 60)
    print("Generating exp4 level sequences...")
    print(f"Variants: {len(VARIANTS)}")
    print(f"Sequences: {NUM_SEQUENCES}")
    print(f"Main levels per sequence: {SEQ_LENGTH}")
    print(f"Min hits per variant: {MIN_HITS_PER_VARIANT}")
    print(f"Total slots: {NUM_SEQUENCES * SEQ_LENGTH}")
    print("=" * 60)
    print()

    seqs = make_sequences(NUM_SEQUENCES, SEQ_LENGTH, MIN_HITS_PER_VARIANT)
    js, usage = format_sequences_js(seqs)

    print()
    print("=" * 60)
    print("Generated sequences (JavaScript format):")
    print("=" * 60)
    print(js)
    print()
    print("=" * 60)
    print("Logistic tracker:")
    print("=" * 60)
    print(logistic_tracker(usage, MIN_HITS_PER_VARIANT))
