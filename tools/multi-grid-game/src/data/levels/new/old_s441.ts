import { LevelConfig } from '../types';

export const old_s441: LevelConfig = {
  id: 'old_s441',
  name: 'old_s441',
  asciiMap:`
WWWWWWWWWWW
WgWWgWWWWWW
W.WW.WWWWWW
W...MWWWWWW
WWWW.WWWWWW
WWWW.WWWWWW
WWWWZWWWWWW
WWWW.WWWWWW
W.........W
W.WW.WW.WBW
WbWWeWWeWGW
WWWWWWWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: {
          path: ["down", "down", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["up", "up", "up", "up", "up"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["down", "down", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["up", "up", "up", "up", "up"],
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
