import { LevelConfig } from '../types';

export const sm611: LevelConfig = {
  id: 'sm611',
  name: 'sm611',
  asciiMap: `
WWWWWWWGWWe
WWWWWWWBWW.
WWWWWWW.WWO
WWWWWW.....
b......WWWW
WWWWWWZ...r
WWWWWW.WWWW
WWWWWW.WWWW
WW.........
WW.WW.WWWRW
WW.WW.WWW.W
WWMWWgWWWgW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["up", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "up", "right", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right", "right", "right", "left", "left", "left", "down", "down", "down", "right", "right", "right", "down", "down", "down"],
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
          path: ["up", "up", "down", "down", "left", "left", "left", "left", "down", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "up", "right", "up", "up", "up"],
          goal: 1,
          type: 'Novice'
        },
        experienced2: {
          path: ["up", "up", "down", "down", "left", "left", "left", "left", "down", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "up", "right", "up", "up", "up"],
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
  stepsRemaining: 105,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
