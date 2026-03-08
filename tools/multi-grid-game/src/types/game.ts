// src/types/game.ts

export interface Position {
  x: number;
  y: number;
}

export interface Agent {
  id: number;
  x: number;
  y: number;
  color: string;
  currentPath?: string[];
  currentMovementIndex: number;
  interactedBy: string[];
  movements: {
    [key: string]: {
      path: string[];
      goal: number;
      type: string;
    };
  };
  observesRemaining: number;
  selectedPath: string | null;
  type?: string;
  isMoving?: boolean;
  isInteracting: boolean;
  resetTrails?: boolean;
  movementHistory?: Array<{x: number, y: number, direction: string, timestamp: number}>;
}

export interface Barrier {
  x: number;
  y: number;
  requiredItems: string[];
  color: string;
}

export interface TreasurePot {
  x: number;
  y: number;
  type: string;
  color: string;
  label: string;
}

export interface Wizard {
  x: number;
  y: number;
  color: string;
  content: string;
  interactedBy: number[];
}

export interface Player {
  x: number;
  y: number;
  isPlayer: boolean;
  id: number;
  isMoving?: boolean;
  resetTrails?: boolean;
}

export interface GameState {
  player: Player;
  agents: Agent[];
  barriers: Barrier[];
  treasurePots: TreasurePot[];
  wizards: Wizard[];
  blocks: Position[];
  inventory: string[];
  p1Inventory: string[];
  p2Inventory: string[];
  openedBarriers: string[];
  stepsRemaining: number;
  goal: {
    type: string;
    description: string;
  };
  gameSessionId?: string;
}