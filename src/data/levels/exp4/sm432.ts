import { LevelConfig } from '../types';

export const sm432: LevelConfig = {
  id: 'sm432',
  name: 'sm432',
  asciiMap: `
WWWWWWWWWgW
eWeWbWWWW.W
.W.W.WWWWBW
O....WWWW.W
.WWWWWWWW.W
..........M
WWW.WWWWWWW
WWW.WWWWWWW
WWW.WWWWWWW
.......Z.rW
BWWWRWWWWWW
GWWWgWWWWWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["left", "left", "left", "left", "up", "up", "up", "up", "left", "left", "left", "up", "up", "right", "right", "right", "right", "up", "up", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right", "left", "left", "left", "left", "down", "down"],
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
          path: ["up", "up", "down", "right", "right", "up", "up", "down", "right", "right", "up", "up", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "down", "down", "down", "down", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced2: {
          path: ["up", "up", "down", "right", "right", "up", "up", "down", "right", "right", "up", "up", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "down", "down", "down", "down", "left", "left", "left", "down", "down"],
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
  stepsRemaining: 135,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
