import { LevelConfig } from '../types';

export const old_s542: LevelConfig = {
  id: 'old_s542',
  name: 'old_s542',
  asciiMap:`
WWbWeWWeWeW
WW.W.WW.W.W
...........
WWW.....WWW
WWWWW.WWWWW
GR...Z....r
WWWWW.WWWWW
WWWWW.WWWWW
WWWWW.WWWWW
M..........
WWWWW.WWWWB
WWWWWgWWWWg
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: {
          path: ["right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["up", "up", "up", "left", "left", "left", "up", "up", "down", "right", "down", "right", "right", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Novice'
        },
        experienced3: {
          path: ["right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["up", "up", "up", "left", "left", "left", "up", "up", "down", "right", "down", "right", "right", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Novice_2'
        },
      }
    },
  },
  stepsRemaining: 95,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
