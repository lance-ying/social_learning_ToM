# sm621.ts Pathing Issues - Diagnostic Report

## Issue Summary
1. **Agent doesn't move toward wizard** - The interact(wizard1) action was not converted to movement
2. **Wrong agent path displayed** - Agent 1 shows agent3's path instead of agent2's path

## Entity Positions in ASCII Map

From `sm621.txt`:
```
bWWWWWWWWWX    row 0: 'b' at (0,0), 'X' at (10,0)
.WWWWWWWWW.    row 1
YWWWWWWWWW.    row 2: 'Y' at (0,2)
...........    row 3
WWWW.WWWWWW    row 4
WWWW.......    row 5
WWWW.WWWWWW    row 6
gWWW.......    row 7: 'g' at (0,7)
BWWW.WWWWWW    row 8: 'B' at (0,8)
M.........e    row 9: 'M' at (0,9), 'e' at (10,9)
BWWW.WWWWWW    row 10: 'B' at (0,10)
GWWW......g    row 11: 'g' at (10,11)
```

### Entity Mapping (top-to-bottom, left-to-right scan):
- **wizard1** = 'b' at (0, 0)
- **wizard2** = 'e' at (10, 9)
- **door1** = 'B' at (0, 8)
- **door2** = 'B' at (0, 10)
- **gem1** = 'g' at (0, 7)
- **gem2** = 'g' at (10, 11)
- **Agent 1 start** = 'X' at (10, 0)
- **Agent 2 start** = 'Y' at (0, 2)

## Exp4 Pathing Data for Scenario2

### agent2 (should map to current Agent 1):
- **Goal**: gem 1 (gem at 0,7)
- **Type**: actual (Expert)
- **Plan Length**: 35 steps
- **Key Actions**:
  ```
  down, down, down, left x10, up, up,
  >>>interact(wizard1)<<<,  # Step 16 - CRITICAL ACTION
  down, down, right x4, down x6, left x4,
  pass(agent2, key1, door1),
  up,
  pickup(agent2, gem1)
  ```

### agent3 (should map to current Agent 2):
- **Goal**: gem 3 (gem at 10,11)
- **Type**: naive (Novice)
- **Plan Length**: 20 steps
- **Plan**:
  ```
  down, right x4, down x7, right x6,
  pickup(agent3, gem3)
  ```

## Current State in sm621.ts

### Agent 1, experienced2:
```typescript
path: ["down", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right"],
goal: 1,
type: 'Expert'
```

**Analysis**: This is exactly agent3's plan (down, 4xright, 7xdown, 6xright) - THE WRONG AGENT'S PATH!

### Agent 2, experienced2:
```typescript
path: [],
goal: 3,
type: 'Novice'
```

**Analysis**: Empty path, but goal is 3 which matches agent3's goal. This seems correct for the novice agent.

## Root Cause Analysis

### Problem 1: Wrong Path Assignment
**What happened**: agent3's plan was converted and assigned to Agent 1 instead of Agent 2

**Why**: The converter likely processed agents in the order they appear in the JSON, and may have confused the agent mapping:
- exp4 agent2 → current Agent 1 ✓ (correct mapping)
- exp4 agent3 → current Agent 2 ✓ (correct mapping)

But the actual paths got swapped somehow during conversion.

### Problem 2: Missing Wizard Interaction
**What should have happened**: When processing agent2's plan, the converter should have:
1. Traced movements: down x3, left x10, up x2 → position (0, 1)
2. Encountered "interact(wizard1)"
3. Called `simple_pathfind((0,1), (0,0), ascii_map)` to get wizard1 position
4. Added resulting path ["up"] to movement_path
5. Updated current_pos to (0, 0)
6. Continued with remaining movements

**What actually happened**: agent2's plan was never converted, so the interact action was never processed.

### Problem 3: Agent Path Display
The user mentioned "agent2 paths are being shown" - this suggests that Agent 2 is displaying movement when it shouldn't. Agent 2 should have an empty path in scenario2 (it's the novice that doesn't move).

## Expected vs Actual for Agent 1, Scenario2

### Expected Path (if agent2's plan was correctly converted):
Starting from X position (10, 0):
1. down x3 → (10, 3)
2. left x10 → (0, 3)
3. up x2 → (0, 1)
4. **interact(wizard1)**: pathfind from (0,1) to (0,0) → ["up"]
5. Now at wizard1 position (0, 0) with item from wizard
6. down x2 → (0, 2)
7. right x4 → (4, 2)
8. down x6 → (4, 8)
9. left x4 → (0, 8)
10. **pass(agent2, key1, door1)**: pathfind to door1 at (0,8) → [] (already there)
11. up x1 → (0, 7)
12. **pickup(agent2, gem1)**: pathfind to gem1 at (0,7) → [] (already there)

**Complete expected path**:
```
["down", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left"]
```

### Actual Path in sm621.ts:
```
["down", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right"]
```

This is agent3's path, which leads to gem3 at (10, 11), not gem1 at (0, 7).

## Verification Check

Let me trace agent3's path from Y position (0, 2):
- Start: (0, 2)
- down: (0, 3)
- right x4: (4, 3)
- down x7: (4, 10)
- right x6: (10, 10)
- This should lead near gem3 at (10, 11) ✓

This confirms agent3's path is what's currently assigned to Agent 1.

## Conclusion

The converter script appears to have mixed up the agent assignments during conversion. The most likely cause is:

1. **Agent Start Position Mismatch**: The converter may have used the wrong start position when converting agent2's plan
2. **Scenario/Agent Iteration Order**: The loop that processes scenarios and agents may have assigned paths incorrectly
3. **Path Update Confusion**: The `update_level_paths()` function may have updated the wrong agent's path in the TypeScript file

To fix this, the converter needs to be re-run with corrected logic, or the paths need to be manually corrected in sm621.ts to:
- Agent 1: Use agent2's converted plan (with interact, pass, pickup actions converted to movements)
- Agent 2: Keep empty path (novice doesn't move)
