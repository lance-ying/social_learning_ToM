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
bWW.WWWWWWZ
WWW.WWWWWW.
...O.......
WWWWW.WWW.W
WWWWW.WWW.W
WWWWWgWWWgW
WWWWWWWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: {
          path: ["down", "down", "left", "left", "left", "left", "left", "left", "left", "up", "up", "up", "up", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["down", "down", "left", "left", "left", "left", "left", "down", "down", "down"],
          goal: 2,
          type: 'Expert'
        },
        experienced3: {
          path: ["down", "down", "left", "left", "left", "left", "left", "left", "left", "up", "up", "up", "up", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["down", "down", "left", "left", "left", "left", "left", "down", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    },
    2: {
      movements: {
        experienced1: {
          path: ["right", "right", "right", "right", "right", "right", "down", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["up", "up", "up", "up", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced3: {
          path: ["right", "right", "right", "right", "right", "right", "down", "down", "down"],
          goal: 3,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["up", "up", "up", "up", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up"],
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
