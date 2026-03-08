from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# /// script
# dependencies = [
#   "dataclasses",
# ]
# ///

@dataclass
class Position:
    x: int
    y: int
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __lt__(self, other):
        if isinstance(other, Position):
            return (self.x, self.y) < (other.x, other.y)
        return NotImplemented
    
    def __eq__(self, other):
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y
        return NotImplemented


@dataclass
class Agent:
    id: int
    position: Position
    color: str


@dataclass
class Treasure:
    position: Position
    treasure_type: str


@dataclass
class Wizard:
    position: Position
    amulet_type: str
    always_has_item: bool
    
    @property
    def content(self) -> str:
        """Backward compatibility property for accessing amulet content"""
        if self.amulet_type == 'none':
            return ''
        return f"{self.amulet_type}Amulet"


@dataclass
class Barrier:
    position: Position
    required_items: List[str]
    color: str


@dataclass
class ParsedMap:
    width: int
    height: int
    walls: List[Position]
    agents: List[Agent]
    treasures: List[Treasure]
    wizards: List[Wizard]
    barriers: List[Barrier]
    player_start: Position


def parse_ascii_map(ascii_map: str) -> ParsedMap:
    """Parse ASCII map string into game elements"""
    lines = ascii_map.strip().split('\n')
    height = len(lines)
    width = max(len(line) for line in lines) if lines else 0
    
    walls = []
    agents = []
    treasures = []
    wizards = []
    barriers = []
    player_start = Position(0, 0)
    treasure_index = 0
    treasure_types = ['A', 'B', 'C']
    
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            pos = Position(x, y)
            
            if char == 'W':
                walls.append(pos)
            elif char == 'g':
                treasures.append(Treasure(pos, treasure_types[treasure_index % len(treasure_types)]))
                treasure_index += 1
            elif char == 'M':
                player_start = pos
            elif char == 'Z':
                agents.append(Agent(1, pos, 'blue'))
            elif char == 'O':
                agents.append(Agent(2, pos, 'green'))
            elif char == 'e':
                wizards.append(Wizard(pos, 'none', False))
            elif char == 'b':
                wizards.append(Wizard(pos, 'blue', False))
            elif char == 'r':  
                wizards.append(Wizard(pos, 'red', True))
            elif char == 'B':
                barriers.append(Barrier(pos, ['blueAmulet'], 'blue'))
            elif char == 'R':
                barriers.append(Barrier(pos, ['redAmulet'], 'red'))
    
    return ParsedMap(
        width=width,
        height=height,
        walls=walls,
        agents=agents,
        treasures=treasures,
        wizards=wizards,
        barriers=barriers,
        player_start=player_start
    )


def is_walkable(pos: Position, parsed_map: ParsedMap, opened_barriers: set = None) -> bool:
    """Check if a position is walkable"""
    if opened_barriers is None:
        opened_barriers = set()
    
    if pos.x < 0 or pos.x >= parsed_map.width or pos.y < 0 or pos.y >= parsed_map.height:
        return False
    
    if pos in parsed_map.walls:
        return False
    
    for barrier in parsed_map.barriers:
        if barrier.position == pos and barrier.position not in opened_barriers:
            return False
    
    return True


def get_neighbors(pos: Position) -> List[Tuple[Position, str]]:
    """Get neighboring positions with movement directions"""
    directions = [
        (Position(pos.x, pos.y - 1), 'up'),
        (Position(pos.x, pos.y + 1), 'down'),
        (Position(pos.x - 1, pos.y), 'left'),
        (Position(pos.x + 1, pos.y), 'right')
    ]
    return directions