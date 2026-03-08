import { LevelConfig } from '../types';

export const sm621: LevelConfig = {
  id: 'sm621',
  name: 'sm621',
  asciiMap: `
bWWWWg.Z...
.WWWWWWWWW.
OWWWWWWWWW.
...........
WW.WWWW.WWW
WW.WWWW.WWW
WW.WWWW.WWW
gW.WWWW.WWW
BW.WWWW.WWW
M.........e
BWWWWWWWWWW
GWWWWWWWWWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "up", "up"],
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
          path: ["up", "up", "down", "down", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "down", "down"],
          goal: 3,
          type: 'Novice'
        },
        experienced2: {
          path: ["up", "up", "down", "down", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "down", "down"],
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
  stepsRemaining: 105,
  goal: {
    type: 'C',
    description: 'Find and obtain Treasure C'
  }
};
