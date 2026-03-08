import { LevelConfig } from '../types';

export const sm511: LevelConfig = {
  id: 'sm511',
  name: 'sm511',
  asciiMap: `
WWWWWWWWWWW
eWWeWWeWWWb
OWW.WW.WWW.
...........
WW.WWWWWWW.
WW.WWWWWWW.
WW.WWWWWWWe
r.ZWWWWWWWW
WW.WWWWWWWW
.......M.Rg
BWWWBWWWWWW
GWWWgWWWWWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["left", "left", "right", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["up", "up", "up", "up", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "down", "down"],
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
          path: ["up", "down", "right", "right", "right", "up", "up", "down", "right", "right", "right", "up", "up", "down", "right", "right", "right", "right", "up", "up", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "down", "down"],
          goal: 3,
          type: 'Novice'
        },
        experienced2: {
          path: ["up", "down", "right", "right", "right", "up", "up", "down", "right", "right", "right", "up", "up", "down", "right", "right", "right", "right", "up", "up", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "down", "down"],
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
