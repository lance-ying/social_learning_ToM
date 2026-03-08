import { LevelConfig } from '../types';

export const levels: LevelConfig = {
  id: 'levels',
  name: 'levels',
  asciiMap: `
sm211, sm221, sm311, sm321, sm331, sm341, sm351, sm361, sm371, sm411, sm421, sm431, sm432, sm511, sm521, sm531, sm541, sm543, sm611, sm612
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: [],
          goal: 1,
          type: 'Novice'
        },
        experienced2: {
          path: [],
          goal: 1,
          type: 'Novice'
        },
        experienced3: {
          path: [],
          goal: 1,
          type: 'Novice'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: [],
          goal: 1,
          type: 'Novice'
        },
        experienced2: {
          path: [],
          goal: 1,
          type: 'Novice'
        },
        experienced3: {
          path: [],
          goal: 1,
          type: 'Novice'
        }
      }
    }
  },
  stepsRemaining: 50,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
