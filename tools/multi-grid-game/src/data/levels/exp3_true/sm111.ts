import { LevelConfig } from '../types';

export const sm111: LevelConfig = {
  id: 'sm111',
  name: 'sm111',
  asciiMap: `
WWWWWWWWWWW
WeWWWrWWWeW
W.WWW.WWW.W
W....X....W
WWWWW.WWWWW
We.......bW
WWWWW.WWWWW
WgBR.MWWWWW
WWWWW.WWWWW
W...Y...BGW
WWWWW.WWWWW
WWWWWgWWWWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["down", "down", "down", "down", "down", "down", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["down", "down", "down", "down", "down", "down", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced3: {
          path: ["down", "down", "down", "down", "down", "down", "down", "down"],
          goal: 3,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["down", "down", "down", "down", "down", "down", "down", "down"],
          goal: 3,
          type: 'Expert_2'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: ["right", "up", "up", "up", "up", "right", "right", "right", "right", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right"],
          goal: 2,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "up", "up", "up", "up", "right", "right", "right", "right", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right"],
          goal: 2,
          type: 'Expert'
        },
        experienced3: {
          path: ["right", "up", "up", "up", "up", "right", "right", "right", "right", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right"],
          goal: 2,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["right", "up", "up", "up", "up", "right", "right", "right", "right", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right"],
          goal: 2,
          type: 'Expert_2'
        }
      }
    }
  },
  stepsRemaining: 85,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
