import { LevelConfig } from '../types';

export const sm612: LevelConfig = {
  id: 'sm612',
  name: 'sm612',
  asciiMap: `
WWWWWWWWeWe
WWWWWWWW.W.
b.........O
WWWWWWZWWWW
WWWWWW.WWWW
GB.....WWWW
WWWWWW.WWWW
WWWWWW.WWWW
r.........M
WW.WWBWWWRW
WW.WW.WWW.W
WW.WWgWWWgW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["up", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "down", "down", "down"],
          goal: 2,
          type: 'Expert'
        },
        experienced2: {
          path: ["down", "down", "down", "down", "down", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "down", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced3: {
          path: [],
          goal: 3,
          type: 'Expert'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: ["up", "up", "down", "left", "left", "up", "up", "down", "left", "left", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Novice'
        },
        experienced2: {
          path: ["up", "up", "down", "left", "left", "up", "up", "down", "left", "left", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Novice'
        },
        experienced3: {
          path: [],
          goal: 1,
          type: 'Novice'
        }
      }
    }
  },
  stepsRemaining: 110,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
