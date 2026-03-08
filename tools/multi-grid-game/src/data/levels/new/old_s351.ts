import { LevelConfig } from '../types';

export const old_s351: LevelConfig = {
  id: 'old_s351',
  name: 'old_s351',
  asciiMap:`
WWWWWWWWWWW
WWWWWeWeWWb
WWWWW.W.WW.
gR..Z......
WWWWW.WWWWW
WWWWW.WWWWW
WWWWW.....r
WWWWW.WWWWW
WWWWW.WWWWW
g.B..M..B.G
WWWWWWWWWWW
WWWWWWWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: {
          path: ["right", "right", "right", "right", "right", "right", "up", "up", "down", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 2,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "down", "down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "up", "up", "up", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Novice'
        },
        experienced3: {
          path: ["right", "right", "right", "right", "right", "right", "up", "up", "down", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 2,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["right", "down", "down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "up", "up", "up", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Novice_2'
        },
      }
    },
  },
  stepsRemaining: 115,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
