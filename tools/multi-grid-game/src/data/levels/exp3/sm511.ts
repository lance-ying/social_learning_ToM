import { LevelConfig } from '../types';

export const sm511: LevelConfig = {
  id: 'sm511',
  name: 'sm511',
  asciiMap: `
eWWeWWeWWWb
.WW.WW.WWW.
.WW.WW.WWW.
.....O.....
WW.WWWWWWW.
r.ZWWWWWWW.
WW.WWWWWWWe
WW.WWWWWWWW
WW.WWWWWWWW
..M......Rg
BWWWBWWWWWW
GWWWgWWWWWW
`.trim(),
  agentPaths: {
    1: {
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
        }
      }
    },
    2: {
      movements: {
        experienced1: {
          path: ["left", "left", "left", "down", "down", "left", "left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced3: {
          path: ["left", "left", "left", "down", "down", "left", "left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "down", "down"],
          goal: 3,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 50,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
