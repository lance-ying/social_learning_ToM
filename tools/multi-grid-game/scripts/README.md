# Scripts Documentation

This directory contains all Python scripts and utilities for the Multi-Grid Game project, organized by function.

## Directory Structure

```
scripts/
├── level_generation/    # Core level generation tools
├── pathfinding/         # Pathfinding algorithms
├── converters/          # Level conversion and path generation scripts
├── archive/             # Historical one-time utility scripts
└── analysis/            # Analysis and visualization tools
```

---

## Level Generation (`level_generation/`)

Core tools for creating and managing level configurations.

### `level_generator.py`
**Purpose:** Converts ASCII map representations to complete TypeScript level configurations.

**Key Features:**
- Parses ASCII maps using `map_parser.py`
- Generates agent paths using pathfinding algorithms
- Creates complete level config with movements, goals, and metadata
- Outputs TypeScript-formatted level files

**Usage:**
```python
from level_generator import LevelGenerator

generator = LevelGenerator()
config = generator.generate_level_config(
    level_id='s211',
    ascii_map=map_string,
    steps_remaining=50,
    goal_type='A',
    goal_description='Find and obtain Treasure A'
)
```

### `map_parser.py`
**Purpose:** Parses ASCII map strings into structured data representations.

**Key Features:**
- Identifies map elements (walls, agents, treasures, wizards, barriers)
- Creates Position objects for all entities
- Validates map structure
- Provides helper functions for map traversal

**Map Symbols:**
- `W` = Wall
- `M` = Player (main agent)
- `Z` = Agent 1
- `X` = Agent 2 (for exp3 two-agent levels)
- `g` = Treasure/Goal
- `e` = Empty wizard space
- `b` = Blue wizard
- `r` = Red wizard
- `B` = Blue barrier
- `R` = Red barrier
- `.` = Empty walkable space

### `level_sequence_gen.py`
**Purpose:** Generates randomized level sequences for experiments with constraints.

**Key Features:**
- Creates balanced sequences with variant distribution
- Enforces exclusivity constraints (same base level variants don't appear together)
- Maintains minimum appearance counts per variant
- Respects Set A/B distance rules
- Always starts with tutorial levels (s111, s112)

**Output Format:**
```javascript
const sequences = [
  ['mod_s111_1', 'mod_s112_1', ...],
  // ... more sequences
];
```

**Constraints:**
- Each sequence starts with s111 and s112
- Each base level has variants (_1, _2)
- Variants of the same base cannot appear in the same sequence
- Exclusive pairs enforced (e.g., s211/s221, s311/s321)

---

## Pathfinding (`pathfinding/`)

Pathfinding algorithms for generating agent movement paths.

### `pathfinder.py`
**Purpose:** A* pathfinding algorithm for multi-agent game mechanics.

**Key Features:**
- Optimal path generation to goals
- Handles wizard interactions (blue/red amulets)
- Manages barrier opening/closing
- Supports multiple agents on same map
- Generates both optimal and suboptimal paths

**Path Types:**
- `experienced1` - Optimal path to goal 1
- `experienced2` - Optimal path to goal 2
- Additional variants for multi-agent scenarios

**Usage:**
```python
from pathfinder import Pathfinder

pathfinder = Pathfinder(parsed_map)
result = pathfinder.find_path(
    start=agent_position,
    goal=treasure_position,
    path_type='experienced1'
)
```

### `pathfinder_for_z_agent.py`
**Purpose:** Specialized pathfinding for Z agent with point cost calculations.

**Key Features:**
- Calculates optimal paths with step costs
- Accounts for wizard interaction costs (5 points)
- Includes buffer points for safety margin (25 points)
- Move cost: 3 points per step
- Generates paths for multiple goal targets

**Point System:**
- Move: -3 points
- Wizard interaction: -5 points
- Buffer: 25 point safety margin

---

## Converters (`converters/`)

Scripts for converting ASCII maps to TypeScript levels and generating paths.

### `convert_exp2_to_levels.py`
**Purpose:** Converts exp2 ASCII maps to TypeScript level files.

**Workflow:**
1. Reads ASCII files from `extracted_ascii_maps/exp2/`
2. Parses map and goal annotations (format: `goal1, goal2`)
3. Generates TypeScript level configuration
4. Outputs to `src/data/levels/exp2/`

**Input Format:**
```
WWWWWWW
W.M..gW
W...g.W
WWWWWWW

B, A
```
(Two goals: B for experienced1, A for experienced2)

**Usage:**
```bash
cd scripts/converters
python convert_exp2_to_levels.py
```

### `convert_exp3_to_levels.py`
**Purpose:** Converts exp3 ASCII maps (two-agent) to TypeScript level files.

**Workflow:**
1. Reads ASCII files from `extracted_ascii_maps/exp3/`
2. Parses map with two agents (Z and X)
3. Parses goal annotations for both agents (format: `Z: goal1, goal2` and `X: goal1, goal2`)
4. Generates TypeScript with paths for both agents
5. Outputs to `src/data/levels/exp3/` with `sm` prefix

**Input Format:**
```
WWWWWWW
W.Z..gW
W.X.g.W
WWWWWWW

Z: B, A
X: A, B
```

**Usage:**
```bash
cd scripts/converters
python convert_exp3_to_levels.py
```

### `generate_paths_for_exp2_levels.py`
**Purpose:** Generates optimal paths for all exp2 levels.

**Workflow:**
1. Reads TypeScript level files from `src/data/levels/exp2/`
2. Extracts ASCII map and goal information
3. Calculates optimal paths for both experienced1 and experienced2
4. Updates TypeScript files with generated paths

**Usage:**
```bash
cd scripts/converters
python generate_paths_for_exp2_levels.py
```

### `generate_paths_for_exp3_levels.py`
**Purpose:** Generates optimal paths for all exp3 (two-agent) levels.

**Workflow:**
1. Reads TypeScript level files from `src/data/levels/exp3/`
2. Extracts ASCII map and goals for both Z and X agents
3. Calculates optimal paths for all agent/goal combinations
4. Updates TypeScript files with generated paths

**Usage:**
```bash
cd scripts/converters
python generate_paths_for_exp3_levels.py
```

---

## Analysis (`analysis/`)

Tools for analyzing and visualizing game data.

### `analyze_sequences.py`
**Purpose:** Analyzes level sequences for pattern distribution and frequency.

**Key Features:**
- Counts occurrences of level variants in sequences
- Identifies usage patterns across all sequences
- Validates sequence balance
- Generates statistics reports

**Usage:**
```bash
cd scripts/analysis
python analyze_sequences.py
```

**Input:** Reads from `src/app/components/game-flow/const_sequences.txt`

**Output:** Console report showing:
- Total occurrences of each level variant
- Unique patterns and their counts
- Distribution statistics

### `create_presentation.py`
**Purpose:** Creates PowerPoint presentations from path visualization images.

**Key Features:**
- Reads visualization images from `visualization/visuals/`
- Creates side-by-side comparisons of experienced1 vs experienced2 paths
- Generates one slide per level
- Automatic layout and formatting

**Dependencies:**
```bash
# Requires python-pptx
pip install python-pptx
```

**Usage:**
```bash
cd scripts/analysis
python create_presentation.py
```

**Output:** `visualization/path_comparison_presentation.pptx`

**Image Naming Convention:**
- Input: `stimuli_s211_agent1_experienced1_and_agent2_experienced1.png`
- Pairs: experienced1 (left) and experienced2 (right)

---

## Archive (`archive/`)

Historical one-time utility scripts kept for reference. These scripts have already been run and served their purpose.

### `convert_old_levels.py`
**Status:** ✅ Completed  
**Purpose:** Converted old JavaScript coordinate-based levels to TypeScript ASCII format.  
**Note:** Used during initial migration from old codebase.

### `extract_ascii_maps.py`
**Status:** ✅ Completed  
**Purpose:** Extracted ASCII maps from TypeScript files and saved as individual text files.  
**Output:** Created files in `extracted_ascii_maps/` directory.

### `fix_imports.py`
**Status:** ✅ Completed  
**Purpose:** Fixed import paths in mod/ directory files.  
**Note:** Contains hardcoded paths from old project structure.

### `rename_files.py`
**Status:** ✅ Completed  
**Purpose:** Renamed `old_s*.ts` files to `mod_s*.ts` format.  
**Note:** Contains hardcoded paths from old project structure.

### `fix_exp2_goal_labels.py`
**Status:** ✅ Completed  
**Purpose:** Fixed goal label assignments in exp2 ASCII maps.  
**Process:** Sorted chests by position (top-to-bottom, left-to-right) and assigned labels A, B, C.

### `update_m_player_points.py`
**Status:** ✅ Completed  
**Purpose:** Updated M player starting points in level files.  
**Calculation:** Based on optimal path costs with buffers.

### `replace_novice_with_expert.sh`
**Status:** ✅ Completed  
**Purpose:** Replaced all instances of "Novice" with "Expert" in exp2 TypeScript files.  
**Format:** Bash script using sed for find/replace.

---

## Common Workflows

### Creating New Levels from ASCII Maps

1. **Create ASCII map file**
   - Place in `extracted_ascii_maps/exp2/` or `extracted_ascii_maps/exp3/`
   - Follow proper format with goal annotations

2. **Convert to TypeScript**
   ```bash
   cd scripts/converters
   # For single-agent levels:
   python convert_exp2_to_levels.py
   
   # For two-agent levels:
   python convert_exp3_to_levels.py
   ```

3. **Generate optimal paths**
   ```bash
   # For exp2 levels:
   python generate_paths_for_exp2_levels.py
   
   # For exp3 levels:
   python generate_paths_for_exp3_levels.py
   ```

4. **Add to index**
   - Import in `src/data/levels/index.ts`
   - Add to levels export object

### Generating Level Sequences

1. **Configure constraints** in `level_sequence_gen.py`
   - Set minimum hits per variant
   - Define exclusive pairs
   - Configure Set A/B rules

2. **Run generator**
   ```bash
   cd scripts/level_generation
   python level_sequence_gen.py
   ```

3. **Copy output** to `src/app/components/game-flow/LevelSequencer.tsx`

### Analyzing Path Visualizations

1. **Generate visualization images** (via frontend debug tools)
   - Save to `visualization/visuals/`
   - Follow naming convention: `stimuli_sXXX_agent1_experiencedN_and_agent2_experiencedN.png`

2. **Create presentation**
   ```bash
   cd scripts/analysis
   python create_presentation.py
   ```

3. **Review** generated PowerPoint at `visualization/path_comparison_presentation.pptx`

---

## Development Notes

### Import Path Changes

After reorganization, if scripts need to import from each other:

```python
# Level generation importing pathfinding
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'pathfinding'))
from pathfinder import Pathfinder

# Or use relative imports if package structure is set up
from ..pathfinding.pathfinder import Pathfinder
```

### Python Environment

Scripts are designed to work with Python 3.7+. Some scripts include inline dependency declarations:

```python
# /// script
# dependencies = [
#   "python-pptx",
# ]
# ///
```

### File Paths

Most scripts use relative paths based on their location. When moving scripts, verify:
- Input directory paths
- Output directory paths
- Import statements

---

## Questions or Issues?

If you need to:
- Add new level types
- Modify pathfinding algorithms
- Change sequence generation rules
- Update archive scripts

Refer to the inline documentation in each script or contact the development team.


