import { LevelConfig } from '../types';

export const sm541: LevelConfig = {
  id: 'sm541',
  name: 'sm541',
  asciiMap: `
WWbWeWWeWeW
WW.W.WW.W.W
....Z......
WWWWW.WWWWW
WWWWW.WWWWW
G.B...WWWWW
WWWWW.WWWWW
MWWWWO....r
.WWWW.WWWWW
...........
.WWWW.WWWWR
.WWWWgWWWWg
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["up", "up", "down", "left", "left", "up", "up", "down", "right", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Novice'
        },
        experienced2: {
          path: ["right", "down", "down", "down", "down", "down", "down", "down", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["up", "up", "down", "left", "left", "up", "up", "down", "right", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Novice'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: ["up", "up", "up", "up", "up", "left", "left", "left", "up", "up", "down", "right", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced3: {
          path: ["up", "up", "up", "up", "up", "left", "left", "left", "up", "up", "down", "right", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        }
      }
    }
  },
  stepsRemaining: 115,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
