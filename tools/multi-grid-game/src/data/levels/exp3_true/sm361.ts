import { LevelConfig } from '../types';

export const sm361: LevelConfig = {
  id: 'sm361',
  name: 'sm361',
  asciiMap: `
eWWWWWWeWWb
.WWWWWW.WW.
.WWWWWW.WW.
....X......
WWWW.WWWWWW
WWWW.WWWWWW
WWWW.WWWWWW
WWWWY.....r
WWWW.WWWWWW
..........M
RWWW.WW.WWW
gWWWgWW.B.G
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["right", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "down", "down", "right", "right", "right"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced3: {
          path: ["right", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "down", "down", "right", "right", "right"],
          goal: 3,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 1,
          type: 'Expert_2'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: ["up", "up", "up", "up", "right", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "down", "down", "right", "right", "right"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["down", "down", "down", "down"],
          goal: 2,
          type: 'Expert'
        },
        experienced3: {
          path: ["up", "up", "up", "up", "right", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "down", "down", "right", "right", "right"],
          goal: 3,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["down", "down", "down", "down"],
          goal: 2,
          type: 'Expert_2'
        }
      }
    }
  },
  stepsRemaining: 160,
  goal: {
    type: 'C',
    description: 'Find and obtain Treasure C'
  }
};
