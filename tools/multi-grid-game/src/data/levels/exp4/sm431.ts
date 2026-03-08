import { LevelConfig } from '../types';

export const sm431: LevelConfig = {
  id: 'sm431',
  name: 'sm431',
  asciiMap: `
WWWWWWWWWgW
eWeWbWWWW.W
.W.W.WWWWBW
O....WWWW.W
.WWWWWWWW.W
.Z.........
WWW.WWWWWWW
WWW.WWWWWWW
WWW.WWWWWWW
.....M...rW
BWWWRWWWWWW
GWWWgWWWWWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["left", "up", "up", "right", "right", "right", "right", "up", "up", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down"],
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
  stepsRemaining: 130,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
