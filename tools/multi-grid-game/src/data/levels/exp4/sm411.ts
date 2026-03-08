import { LevelConfig } from '../types';

export const sm411: LevelConfig = {
  id: 'sm411',
  name: 'sm411',
  asciiMap: `
WWWWWWWWWWW
eWeWeWWWWWb
OW.W.WWWWW.
...........
WW.WWWWWWWW
WWZWWWWWWWW
WW.WWWMWWWW
WW.WWW.WWWW
WW.WWW.WWWW
.........Rg
BW.WBWWWWWW
GWrWgWWWWWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["down", "down", "down", "down", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["up", "up", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "down", "down"],
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
          path: ["up", "down", "right", "right", "up", "up", "down", "right", "right", "up", "up", "down", "right", "right", "right", "right", "right", "right", "up", "up", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "down", "down"],
          goal: 3,
          type: 'Novice'
        },
        experienced2: {
          path: ["up", "down", "right", "right", "up", "up", "down", "right", "right", "up", "up", "down", "right", "right", "right", "right", "right", "right", "up", "up", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "down", "down"],
          goal: 3,
          type: 'Novice'
        },
        experienced3: {
          path: [],
          goal: 3,
          type: 'Novice'
        }
      }
    }
  },
  stepsRemaining: 150,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
