import { LevelConfig } from '../types';

export const sm221: LevelConfig = {
  id: 'sm221',
  name: 'sm221',
  asciiMap: `
WWbWWWWWGWW
WW.WWWWWBWW
WW.WWWWW.WW
.........WW
WWWWW.WW.WW
WWWWW.WW.WW
WWWWW.WW.WW
gWWWW.WW.WW
RWWWW.WW.WW
...M.....Oe
RWW.WWWWWWW
gWW...Z...r
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["left", "left", "left", "up", "up", "right", "right", "up", "up", "up", "up", "up", "up", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "right", "right", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "up", "up", "left", "left", "left", "down", "down"],
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
          path: ["right", "left", "left", "left", "left", "up", "up", "up", "up", "up", "up", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "right", "right", "up", "up", "up"],
          goal: 1,
          type: 'Novice'
        },
        experienced2: {
          path: ["right", "left", "left", "left", "left", "up", "up", "up", "up", "up", "up", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "right", "right", "up", "up", "up"],
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
  stepsRemaining: 95,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
