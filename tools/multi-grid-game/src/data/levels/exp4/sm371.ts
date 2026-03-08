import { LevelConfig } from '../types';

export const sm371: LevelConfig = {
  id: 'sm371',
  name: 'sm371',
  asciiMap: `
WWWWWWWWWGW
eWWWWWWWWBW
.WWWWWWWW.W
M.........W
.WW.WWWWWWW
bWW.WWWWWWO
WWW.WWWWWW.
...........
.WWWW.WWW.W
.WWWW.WWW.W
ZWWWW.WWW.W
WWWWWgWWWgW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["up", "up", "up", "right", "right", "right", "up", "up", "up", "up", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["up", "up", "up", "right", "right", "right", "up", "up", "up", "up", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced3: {
          path: [],
          goal: 1,
          type: 'Expert'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: ["down", "down", "left", "left", "left", "left", "left", "down", "down", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced2: {
          path: ["down", "down", "left", "left", "left", "left", "left", "left", "left", "up", "up", "up", "up", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up"],
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
  stepsRemaining: 65,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
