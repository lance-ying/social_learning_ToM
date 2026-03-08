import { LevelConfig } from '../types';

export const sm441: LevelConfig = {
  id: 'sm441',
  name: 'sm441',
  asciiMap: `
WgWWgWWWWWW
W.WW.WWWWWW
W.WW.WWWWWW
W...MWWWWWW
WWWW.WWWWWW
WWWW.WWWWWW
e...ZWWWWWW
WWWW.WWWWWW
WWWW.WWWWWW
W.O.......W
W.WW.WW.WBW
WbWWeWWeWGW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: {
          path: ["down", "down", "down", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["up", "up", "up", "up", "up", "up"],
          goal: 2,
          type: 'Expert'
        },
        experienced3: {
          path: ["down", "down", "down", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["up", "up", "up", "up", "up", "up"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    },
    2: {
      movements: {
        experienced1: {
          path: ["right", "right", "up", "up", "up", "up", "up", "up", "left", "left", "left", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced3: {
          path: ["right", "right", "up", "up", "up", "up", "up", "up", "left", "left", "left", "up", "up", "up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 50,
  goal: {
    type: 'C',
    description: 'Find and obtain Treasure C'
  }
};
