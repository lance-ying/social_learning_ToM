// src/data/mapData.ts
import { Agent, GameState } from '../types/game';

export type AsciiMapSymbol = 'W' | 'g' | 'G' | 'O' | 'Z' | 'X' | 'Y' | 'M' | 'E' | 'b' | 'r' | 'B' | 'R' | '.';

const emptyTemplates = {
  treasurePots: [] as Array<{
    x: number;
    y: number;
    type: string;
    color: string;
    label: string;
  }>,
  wizards: [] as Array<{
    x: number;
    y: number;
    content: string;
    color: string;
    interacted: boolean;
    interactedBy: number[];
  }>,
  agents: [] as Array<{
    id: number;
    x: number;
    y: number;
    color: string;
    currentMovementIndex: number;
    movements: {
      experienced1: { path: string[]; goal: number; type: string };
      experienced2: { path: string[]; goal: number; type: string };
    };
    observesRemaining: number;
    selectedPath: string | null;
    resetTrails?: boolean;
  }>,
  barriers: [] as Array<{
    x: number;
    y: number;
    requiredItems: string[];
    color: string;
  }>
};

export function getMapDimensions(asciiMap: string): { width: number; height: number } {
  const rows = asciiMap.trim().split('\n');
  const height = rows.length;
  const width = Math.max(...rows.map(row => row.length));
  return { width, height };
}

export function parseAsciiMap(asciiMap: string): Partial<GameState> {
  const rows = asciiMap.trim().split('\n');
  const blocks: {x: number, y: number}[] = [];
  const treasurePots = [...emptyTemplates.treasurePots];
  const wizards = [...emptyTemplates.wizards];
  const agents = [...emptyTemplates.agents];
  const barriers = [...emptyTemplates.barriers];
  let player = { x: 0, y: 0 };
  let treasureIndex = 0;
  const treasureTypes = ['A', 'B', 'C'];

  rows.forEach((row, y) => {
    [...row].forEach((char, x) => {
      switch (char) {
        case 'W':
          blocks.push({ x, y });
          break;
        case 'g':
        case 'G':
          treasurePots.push({
            x, y,
            type: treasureTypes[treasureIndex++],
            color: 'text-yellow-500',
            label: treasureTypes[treasureIndex - 1]
          });
          break;
        case 'e':
          wizards.push({
            x, y,
            content: 'nothing',
            color: 'text-blue-500',
            interacted: false,
            interactedBy: []
          });
          break;
        case 'b':
          wizards.push({
            x, y,
            content: 'blueAmulet',
            color: 'text-blue-500',
            interacted: false,
            interactedBy: []
          });
          break;
        case 'r':
          wizards.push({
            x, y,
            content: 'redAmulet',
            color: 'text-red-500',
            interacted: false,
            interactedBy: []
          });
          break;
        case 'B':
          barriers.push({
            x, y,
            requiredItems: ["blueAmulet"],
            color: 'text-blue-500'
          });
          break;
        case 'R':
          barriers.push({
            x, y,
            requiredItems: ["redAmulet"],
            color: 'text-red-500'
          });
          break;
        case 'M':
          player = { x, y };  // M is the player
          break;
        case 'Z':
          agents.push({
            id: 2,
            x, y,
            color: 'blue',
            currentMovementIndex: 0,
            observesRemaining: 25,
            selectedPath: null,
            resetTrails: false,
            movements: {
              experienced1: { path: [], goal: 1, type: 'Expert' },
              experienced2: { path: [], goal: 2, type: 'Novice' }
            }
          });
          break;
        case 'O':
          agents.push({
            id: 3,
            x, y,
            color: 'green',
            currentMovementIndex: 0,
            observesRemaining: 25,
            selectedPath: null,
            resetTrails: false,
            movements: {
              experienced1: { path: [], goal: 1, type: 'Expert' },
              experienced2: { path: [], goal: 2, type: 'Novice' }
            }
          });
          break;
        case 'X':
          agents.push({
            id: 2,
            x, y,
            color: 'blue',
            currentMovementIndex: 0,
            observesRemaining: 25,
            selectedPath: null,
            resetTrails: false,
            movements: {
              experienced1: { path: [], goal: 1, type: 'Expert' },
              experienced2: { path: [], goal: 2, type: 'Novice' }
            }
          });
          break;
        case 'Y':
          agents.push({
            id: 3,
            x, y,
            color: 'green',
            currentMovementIndex: 0,
            observesRemaining: 25,
            selectedPath: null,
            resetTrails: false,
            movements: {
              experienced1: { path: [], goal: 1, type: 'Expert' },
              experienced2: { path: [], goal: 2, type: 'Novice' }
            }
          });
          break;
      }
    });
  });

  return {
    blocks,
    treasurePots,
    wizards,
    agents: agents as unknown as Agent[],
    barriers,
    player: { ...player, isPlayer: true, id: 0 }
  };
}