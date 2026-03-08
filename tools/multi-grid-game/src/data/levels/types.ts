import { GameState } from '../../types/game';

export interface LevelConfig {
  id: string;
  name: string;
  asciiMap: string;
  agentPaths: {
    [key: string]: {
      movements: {
        experienced1: { path: string[]; goal: number; type: string };
        experienced2: { path: string[]; goal: number; type: string };
        experienced3?: { path: string[]; goal: number; type: string };
        experienced4?: { path: string[]; goal: number; type: string };
      };
    };
  };
  stepsRemaining: number;
  goal: {
    type: string;
    description: string;
  };
} 