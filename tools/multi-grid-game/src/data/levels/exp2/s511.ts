import { LevelConfig } from '../types';

export const s511: LevelConfig = {
  id: 's511',
  name: 's511',
  asciiMap:`
eWWeWWeWWWb
.WW.WW.WWW.
.WW.WW.WWW.
...........
WW.WWWWWWW.
r.ZWWWWWWW.
WW.WWWWWWWe
WWMWWWWWWWW
WW.WWWWWWWW
.........Rg
BWWWBWWWWWW
GWWWgWWWWWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["up", "up", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "down", "down"],
          goal: 2,
          type: 'Expert'
        },
        experienced2: {
          path: ["left", "left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced3: {
          path: ["up", "up", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "down", "down"],
          goal: 2,
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
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
