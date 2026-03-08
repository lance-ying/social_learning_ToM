import { LevelConfig } from '../types';

export const sm541: LevelConfig = {
  id: 'sm541',
  name: 'sm541',
  asciiMap: `
WWbWeWWeWeW
WW.W.WW.W.W
...........
WWWWWXWWWWW
WWWWW.WWWWW
G.B...WWWWW
WWWWW.WWWWW
MWWWWY....r
.WWWW.WWWWW
...........
.WWWW.WWWWR
.WWWWgWWWWg
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["up", "left", "left", "left", "up", "up", "down", "right", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["down", "down", "down", "down", "down", "down", "down", "down"],
          goal: 2,
          type: 'Expert'
        },
        experienced3: {
          path: ["up", "left", "left", "left", "up", "up", "down", "right", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["down", "down", "down", "down", "down", "down", "down", "down"],
          goal: 2,
          type: 'Expert_2'
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
          type: 'Expert_2'
        },
        experienced4: {
          path: ["right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Expert_2'
        }
      }
    }
  },
  stepsRemaining: 125,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
