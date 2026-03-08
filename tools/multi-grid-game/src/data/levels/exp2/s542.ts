import { LevelConfig } from '../types';

export const s542: LevelConfig = {
  id: 's542',
  name: 's542',
  asciiMap:`
We...bWWWWW
WWWZWWWWWWW
...........
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
    2: {
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
        },
      }
    },
  },
  stepsRemaining: 120,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
