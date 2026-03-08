import { LevelConfig } from '../types';

export const sm341: LevelConfig = {
  id: 'sm341',
  name: 'sm341',
  asciiMap: `
WWWWWeWrWgW
eWWWW.W.WRW
.WWWW.W.W.W
O......Z..W
.W.WWWW.WWW
.W.WWWW.WWW
bW.WWWW.WWW
WW.WWWW.WWW
M..........
WWWWW.WW.WW
WWWWWBWW.WW
WWWWWGWWgWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["up", "up", "up", "down", "down", "right", "right", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "up", "up", "right", "right", "down", "down", "down", "down", "down", "right", "right", "right", "down", "down", "down"],
          goal: 2,
          type: 'Expert'
        },
        experienced3: {
          path: [],
          goal: 2,
          type: 'Expert'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: ["up", "up", "down", "down", "down", "down", "up", "up", "right", "right", "down", "down", "down", "down", "down", "right", "right", "right", "down", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced2: {
          path: ["up", "up", "down", "down", "down", "down", "up", "up", "right", "right", "down", "down", "down", "down", "down", "right", "right", "right", "down", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: [],
          goal: 2,
          type: 'Novice'
        }
      }
    }
  },
  stepsRemaining: 105,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
