import { LevelConfig } from '../types';

export const sm331: LevelConfig = {
  id: 'sm331',
  name: 'sm331',
  asciiMap: `
WWWeWeWWWWG
WWW.W.WWWWB
b.O........
WWW.WWWWWWW
WWW.WWWWWWW
WWW.WWWWWWW
WWW.WWWWWWW
........WWW
RWW.WWW.WWW
.WW.WWWZ..r
.WW.WWW.WWW
gWWMWWW..Bg
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["right", "right", "right", "left", "left", "up", "up", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down"],
          goal: 2,
          type: 'Expert'
        },
        experienced2: {
          path: ["up", "up", "left", "left", "left", "left", "up", "up", "up", "up", "up", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced3: {
          path: [],
          goal: 1,
          type: 'Expert'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: ["left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up"],
          goal: 1,
          type: 'Novice'
        },
        experienced2: {
          path: ["left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up"],
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
  stepsRemaining: 90,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
