import { LevelConfig } from '../types';

export const sm211: LevelConfig = {
  id: 'sm211',
  name: 'sm211',
  asciiMap: `
WWWWWWWWWWW
eWWWWWWWWWb
OWWWWWWWWW.
...........
WW.WWWWWWWW
WW.WWWWWWWW
WW.WWWWWWWW
WW.WWWWWWWW
WW.WWWWWWWW
........MRg
BWWWBW.WWWW
GWWWgWZ...r
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["right", "right", "right", "right", "left", "left", "left", "up", "up", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["up", "up", "left", "left", "left", "left", "up", "up", "up", "up", "up", "up", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "down", "down"],
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
          path: ["up", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "down", "down"],
          goal: 3,
          type: 'Novice'
        },
        experienced2: {
          path: ["up", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "down", "down"],
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
  stepsRemaining: 145,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
