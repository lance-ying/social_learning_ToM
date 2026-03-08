import { LevelConfig } from '../types';

export const sm321: LevelConfig = {
  id: 'sm321',
  name: 'sm321',
  asciiMap: `
WWbWWWWWWWW
WW.WWWWWWWW
WW.WWWWWWWW
.........WW
WWWWWWWW.WW
WWWWWWWW.WW
WWWWWWWW.WW
gWWWWWWW..e
BWWWWWWW.WW
...M....O.e
BWW.WWWWWWW
GWW.....Z.g
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["left", "left", "left", "left", "left", "up", "up", "right", "right", "right", "right", "right", "up", "up", "up", "up", "up", "up", "left", "left", "left", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right"],
          goal: 3,
          type: 'Expert'
        },
        experienced3: {
          path: [],
          goal: 3,
          type: 'Expert'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: ["right", "right", "left", "up", "up", "right", "right", "left", "up", "up", "up", "up", "left", "left", "left", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "up", "up"],
          goal: 1,
          type: 'Novice'
        },
        experienced2: {
          path: ["right", "right", "left", "up", "up", "right", "right", "left", "up", "up", "up", "up", "left", "left", "left", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "up", "up"],
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
  stepsRemaining: 155,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
