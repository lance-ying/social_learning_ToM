import { LevelConfig } from '../types';

export const s442: LevelConfig = {
  id: 's442',
  name: 's442',
  asciiMap:`
WgWWgWWWWWW
W.WW.WWWWWW
W.WW.WWWWWW
W...Z.....e
WWWW.WWWWWW
WWWW.WWWWWW
WWWWMWWWWWW
WWWW.WWWWWW
W.........W
W.WW.WW.WBW
W.WW.WW.W.W
WbWWeWWeWGW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["down", "down", "down", "down", "down", "left", "left", "left", "down", "down", "down", "up", "up", "right", "right", "right", "right", "right", "right", "right", "right", "down", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["up", "up", "up"],
          goal: 2,
          type: 'Expert'
        },
        experienced3: {
          path: ["down", "down", "down", "down", "down", "left", "left", "left", "down", "down", "down", "up", "up", "right", "right", "right", "right", "right", "right", "right", "right", "down", "down", "down"],
          goal: 3,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["up", "up", "up"],
          goal: 2,
          type: 'Novice_2'
        },
      }
    },
  },
  stepsRemaining: 90,
  goal: {
    type: 'C',
    description: 'Find and obtain Treasure C'
  }
};
