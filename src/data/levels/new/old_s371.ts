import { LevelConfig } from '../types';

export const old_s371: LevelConfig = {
  id: 'old_s371',
  name: 'old_s371',
  asciiMap:`
WWWWWWWWWGW
eWWWWWWWWBW
.WWWWWWWW.W
M.........W
.WW.WWWWWWW
bWW.WWWWWWW
WWW.WWWWWWW
...Z.......
WWWWW.WWW.W
WWWWW.WWW.W
WWWWWgWWWgW
WWWWWWWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: {
          path: ["up", "up", "up", "up", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right", "down", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["up", "up", "up", "up", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["right", "right", "down", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        },
      }
    },
  },
  stepsRemaining: 70,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
