import { LevelConfig } from '../types';

export const sm543: LevelConfig = {
  id: 'sm543',
  name: 'sm543',
  asciiMap: `
We.M.bWWWWW
WWW.WWWWWWW
...........
WWWWW.WWWWW
WWWWW.WWWWW
G.B.......r
WWWWW.WWWWW
WWWWW.WWWWW
WWWWW.WWWWW
Y.........X
WWWWW.WWWWR
WWWWWgWWWWg
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["left", "left", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Expert'
        },
        experienced2: {
          path: ["left", "left", "left", "left", "left", "up", "up", "up", "up", "up", "up", "up", "left", "left", "up", "up", "right", "right", "left", "down", "down", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced3: {
          path: ["left", "left", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["left", "left", "left", "left", "left", "up", "up", "up", "up", "up", "up", "up", "left", "left", "up", "up", "right", "right", "left", "down", "down", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert_2'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: ["right", "right", "right", "right", "right", "up", "up", "up", "up", "up", "up", "up", "left", "left", "up", "up", "right", "right", "left", "down", "down", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right", "right", "right", "right", "up", "up", "up", "up", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced3: {
          path: ["right", "right", "right", "right", "right", "up", "up", "up", "up", "up", "up", "up", "left", "left", "up", "up", "right", "right", "left", "down", "down", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["right", "right", "right", "right", "right", "up", "up", "up", "up", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Expert_2'
        }
      }
    }
  },
  stepsRemaining: 75,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
