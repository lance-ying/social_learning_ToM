#!/usr/bin/env python3
"""
Analyze const sequences from const_sequences.txt
1. Count mod_sXXX_1 patterns in sequences
2. Count mod_sXXX_ascii instances in JSON
"""

import re
import json
from collections import Counter

def main():
    # Read the file
    with open('src/app/components/game-flow/const_sequences.txt', 'r') as f:
        content = f.read()

    # Extract the sequences array (lines 1-11)
    sequences_section = content.split('];')[0] + ']'

    # Extract all mod_sXXX_1 patterns
    mod_1_pattern = re.findall(r"'(mod_s\d+_1)'", sequences_section)

    # Count occurrences
    mod_1_counter = Counter(mod_1_pattern)

    # Extract the JSON object (line 14)
    json_match = re.search(r'\{.*\}', content)
    if json_match:
        ascii_data = json.loads(json_match.group())
    else:
        ascii_data = {}

    # Print results
    print("=" * 60)
    print("ANALYSIS OF CONST SEQUENCES")
    print("=" * 60)

    print("\n1. mod_sXXX_1 PATTERNS IN SEQUENCES:")
    print("-" * 60)
    print(f"Total occurrences: {len(mod_1_pattern)}")
    print(f"Unique patterns: {len(mod_1_counter)}")
    print("\nBreakdown by pattern:")
    for pattern, count in sorted(mod_1_counter.items()):
        print(f"  {pattern}: {count} times")

    print("\n" + "=" * 60)
    print("2. mod_sXXX_ascii INSTANCES:")
    print("-" * 60)
    print(f"Total unique ascii patterns: {len(ascii_data)}")
    print(f"Total count (sum of all values): {sum(ascii_data.values())}")
    print("\nBreakdown by pattern:")
    for pattern, count in sorted(ascii_data.items()):
        print(f"  {pattern}: {count}")

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("-" * 60)
    print(f"mod_sXXX_1 total hits: {len(mod_1_pattern)}")
    print(f"mod_sXXX_ascii total count: {sum(ascii_data.values())}")
    print("=" * 60)

if __name__ == "__main__":
    main()
