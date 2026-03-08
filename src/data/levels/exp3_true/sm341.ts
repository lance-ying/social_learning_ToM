import { LevelConfig } from '../types';

export const sm341: LevelConfig = {
  id: 'sm341',
  name: 'sm341',
  asciiMap: `
eWWWWrWWWgW
.WWWW.WWWRW
.WWWW.WWW.W
..Y.X.....W
.WWWWWW.WWW
.WWWWWW.WWW
bWWWWWW.WWW
WWWWWWW.WWW
...........
.WWWW.WWW.W
.WWWWBWWW.W
MWWWWGWWWgW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["right", "right", "right", "down", "down", "down", "down", "down", "right", "right", "down", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["left", "left", "left", "left", "down", "down", "down", "up", "up", "right", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "down", "left", "left", "down", "down", "down"],
          goal: 2,
          type: 'Expert'
        },
        experienced3: {
          path: ["right", "right", "right", "down", "down", "down", "down", "down", "right", "right", "down", "down", "down"],
          goal: 3,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["left", "left", "left", "left", "down", "down", "down", "up", "up", "right", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "down", "left", "left", "down", "down", "down"],
          goal: 2,
          type: 'Expert_2'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: ["left", "left", "down", "down", "down", "up", "up", "right", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "down", "left", "left", "down", "down", "down"],
          goal: 2,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right", "right", "up", "up", "up", "down", "down", "right", "right", "right", "right", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced3: {
          path: ["left", "left", "down", "down", "down", "up", "up", "right", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "down", "left", "left", "down", "down", "down"],
          goal: 2,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["right", "right", "right", "up", "up", "up", "down", "down", "right", "right", "right", "right", "up", "up", "up"],
          goal: 1,
          type: 'Expert_2'
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
