import { LevelConfig } from '../types';

export const sm544: LevelConfig = {
  id: 'sm544',
  name: 'sm544',
  asciiMap: `
WWWWbWeWWWW
WWWW.M.WWWW
...........
WWWWW.WWWWW
WWWWW.WWWWW
G.B.......r
WWWWW.WWWWW
WWWWW.WWWWW
WWWWW.WWWWW
Z.........O
WWWWW.WWWWR
WWWWWgWWWWg
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: {
          path: ["right", "right", "right", "right", "right", "up", "up", "up", "up", "up", "up", "up", "up", "left", "up", "down", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right", "right", "right", "right", "up", "up", "up", "up", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced3: {
          path: ["right", "right", "right", "right", "right", "up", "up", "up", "up", "up", "up", "up", "up", "left", "up", "down", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["right", "right", "right", "right", "right", "up", "up", "up", "up", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Novice_2'
        }
      }
    },
    2: {
      movements: {
        experienced1: {
          path: ["left", "left", "left", "left", "left", "up", "up", "up", "up", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["left", "left", "left", "left", "left", "up", "up", "up", "up", "up", "up", "up", "up", "left", "up", "down", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced3: {
          path: ["left", "left", "left", "left", "left", "up", "up", "up", "up", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["left", "left", "left", "left", "left", "up", "up", "up", "up", "up", "up", "up", "up", "left", "up", "down", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 50,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
