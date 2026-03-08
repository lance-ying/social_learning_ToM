import { LevelConfig } from '../types';

export const old_s311: LevelConfig = {
  id: 'old_s311',
  name: 'old_s311',
  asciiMap:`
eWWeWW.WWWb
.WW.WW.WWW.
.WW.WW.WWW.
...........
WW.WWWWWWW.
r.ZWWWWWWW.
WW.WWWWWWW.
WWMWWWWWWWW
WW.WWWWWWWW
.........Rg
BWWWBWWWWWW
GWWWgWWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: {
          path: ["up", "up", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["left", "left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Novice'
        },
        experienced3: {
          path: ["up", "up", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "down", "down"],
          goal: 3,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["left", "left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Novice_2'
        },
      }
    },
  },
  stepsRemaining: 130,
  goal: {
    type: 'C',
    description: 'Find and obtain Treasure C'
  }
};
