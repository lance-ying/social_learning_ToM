import { LevelConfig } from '../types';

export const sm432: LevelConfig = {
  id: 'sm432',
  name: 'sm432',
  asciiMap: `
eWeWbWeWWgW
.W.W.W.WWBW
.W.W.W.WW.W
X......WW.W
.WWWWWWWW.W
...M.......
WWW.WWWWWWW
WWW.WWWWWWW
WWW.WWWWWWW
.....Y...rW
BWWWRWWWWWW
GWWWgWWWWWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["down", "down", "right", "right", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced3: {
          path: ["down", "down", "right", "right", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down"],
          goal: 3,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "up", "up"],
          goal: 1,
          type: 'Expert_2'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: ["right", "right", "right", "right", "left", "left", "left", "left", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["left", "left", "up", "up", "up", "up", "left", "left", "left", "up", "up", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "down", "down", "down", "down", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Expert'
        },
        experienced3: {
          path: ["right", "right", "right", "right", "left", "left", "left", "left", "down", "down"],
          goal: 3,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["left", "left", "up", "up", "up", "up", "left", "left", "left", "up", "up", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "down", "down", "down", "down", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Expert_2'
        }
      }
    }
  },
  stepsRemaining: 130,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
