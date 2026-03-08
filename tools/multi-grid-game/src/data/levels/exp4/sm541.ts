import { LevelConfig } from '../types';

export const sm541: LevelConfig = {
  id: 'sm541',
  name: 'sm541',
  asciiMap: `
WWbWeWWeWWW
WW.W.WW.WWW
....O......
WWWWW.WWWWW
WWWWW.WWWWW
G.B..Z....r
WWWWW.WWWWW
MWWWW.WWWWW
.WWWW.WWWWW
...........
WWWWW.WWWWR
WWWWWgWWWWg
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["up", "up", "up", "left", "left", "left", "up", "up", "down", "right", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
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
          path: ["up", "up", "down", "left", "left", "up", "up", "down", "right", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Novice'
        },
        experienced2: {
          path: ["up", "up", "down", "left", "left", "up", "up", "down", "right", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
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
  stepsRemaining: 115,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
