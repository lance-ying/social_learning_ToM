import { LevelConfig } from '../types';

export const s441: LevelConfig = {
  id: 's441',
  name: 's441',
  asciiMap:`
WgWWgWWWWWW
W.WW.WWWWWW
W.WW.WWWWWW
W...MWWWWWW
WWWW.WWWWWW
WWWW.WWWWWW
e...ZWWWWWW
WWWW.WWWWWW
WWWW.WWWWWW
W.........W
W.WW.WW.WBW
WbWWeWWeWGW
`.trim(),
  agentPaths: {
    2: {
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
        },
      }
    },
  },
  stepsRemaining: 95,
  goal: {
    type: 'C',
    description: 'Find and obtain Treasure C'
  }
};
