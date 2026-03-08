import { LevelConfig } from '../types';

export const sm421: LevelConfig = {
  id: 'sm421',
  name: 'sm421',
  asciiMap: `
WWbWWWWWWWW
WW.WWWWWWWW
WW.WWWWWWWW
.........WW
WWWWWWWW.WW
WWWWWWWW..e
WWWWWWWW.WW
gWWWWWWW..e
BWWWWWWW.WW
M....Z..O.e
BWW.WWWWWWW
GWW.......g
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["right", "right", "right", "up", "up", "up", "up", "up", "up", "left", "left", "left", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["left", "left", "down", "down", "right", "right", "right", "right", "right", "right", "right"],
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
          path: ["right", "right", "left", "up", "up", "right", "right", "left", "up", "up", "right", "right", "left", "up", "up", "left", "left", "left", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "up", "up"],
          goal: 1,
          type: 'Novice'
        },
        experienced2: {
          path: ["right", "right", "left", "up", "up", "right", "right", "left", "up", "up", "right", "right", "left", "up", "up", "left", "left", "left", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "up", "up"],
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
  stepsRemaining: 165,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
