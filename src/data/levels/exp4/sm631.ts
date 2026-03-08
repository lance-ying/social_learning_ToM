import { LevelConfig } from '../types';

export const sm631: LevelConfig = {
  id: 'sm631',
  name: 'sm631',
  asciiMap: `
bWWWWWWWWWZ
.WWWWWWWWW.
.WWWWWWWWW.
...........
WWWW.WWWWWW
WWWW.......
WWWW.WWWWWW
gWWW.......
BWWW.WWWWWW
M......O..e
BWWW.WWWWWW
GWWW......g
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["down", "down", "down", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["down", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced3: {
          path: ["down", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "up", "up"],
          goal: 1,
          type: 'Expert'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: ["right", "right", "right", "left", "left", "left", "left", "left", "up", "up", "up", "up", "up", "up", "left", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced2: {
          path: ["left", "left", "left", "down", "down", "right", "right", "right", "right", "right", "right"],
          goal: 3,
          type: 'Novice'
        },
        experienced3: {
          path: ["right", "right", "right", "left", "left", "left", "left", "left", "up", "up", "up", "up", "up", "up", "left", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice'
        }
      }
    }
  },
  stepsRemaining: 125,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
