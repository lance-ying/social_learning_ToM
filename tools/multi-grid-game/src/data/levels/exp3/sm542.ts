import { LevelConfig } from '../types';

export const sm542: LevelConfig = {
  id: 'sm542',
  name: 'sm542',
  asciiMap: `
We...bWWWWW
WWWZWWWWWWW
O..........
WWWWW.WWWWW
WWWWW.WWWWW
G.B.......r
WWWWW.WWWWW
WWWWW.WWWWW
WWWWW.WWWWW
M..........
WWWWW.WWWWR
WWWWWgWWWWg
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: {
          path: ["up", "right", "right", "left", "down", "down", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["down", "right", "right", "down", "down", "down", "down", "down", "down", "down", "down", "down"],
          goal: 2,
          type: 'Expert'
        },
        experienced3: {
          path: ["up", "right", "right", "left", "down", "down", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["down", "right", "right", "down", "down", "down", "down", "down", "down", "down", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    },
    2: {
      movements: {
        experienced1: {
          path: ["right", "right", "right", "right", "right", "down", "down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right", "right", "up", "up", "right", "right", "left", "down", "down", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced3: {
          path: ["right", "right", "right", "right", "right", "down", "down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["right", "right", "right", "up", "up", "right", "right", "left", "down", "down", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Novice_2'
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
