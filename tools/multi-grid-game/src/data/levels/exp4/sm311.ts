import { LevelConfig } from '../types';

export const sm311: LevelConfig = {
  id: 'sm311',
  name: 'sm311',
  asciiMap: `
WWWWWWWWWWW
eWeWWWWWWWb
OW.WWWWWWW.
...........
WW.WWWWWWWW
WWZWWWWWWWW
WW.WWWWWWWW
WW.WWWWWWWW
WW.WWWWWWWW
.......M.Rg
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
          path: ["up", "down", "right", "right", "up", "up", "down", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "down", "down"],
          goal: 3,
          type: 'Novice'
        },
        experienced2: {
          path: ["up", "down", "right", "right", "up", "up", "down", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "down", "down"],
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
  stepsRemaining: 140,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
