import { LevelConfig } from '../types';

export const sm351: LevelConfig = {
  id: 'sm351',
  name: 'sm351',
  asciiMap: `
WWWWWeWeWWb
WWWWW.W.WW.
WWWWW.W.WW.
gR..Z......
WWWWW.WWWWW
WWWWWOWWWWW
WWWWW.....r
WWWWW.WWWWW
WWWWW.WWWWW
g.B.....B.G
WWWWW.WWWWW
WWWWWMWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: {
          path: ["right", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 2,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "down", "down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "up", "up", "up", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced3: {
          path: ["right", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 2,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["right", "down", "down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "up", "up", "up", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Novice_2'
        }
      }
    },
    2: {
      movements: {
        experienced1: {
          path: ["down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "up", "up", "up", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["up", "up", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right"],
          goal: 3,
          type: 'Expert'
        },
        experienced3: {
          path: ["down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "up", "up", "up", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["up", "up", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right"],
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
