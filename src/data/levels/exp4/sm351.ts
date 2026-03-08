import { LevelConfig } from '../types';

export const sm351: LevelConfig = {
  id: 'sm351',
  name: 'sm351',
  asciiMap: `
WWWWWWWWWWW
WWWWWeWeWWb
WWWWW.W.WW.
gR...O.....
WWWWW.WWWWW
WWWWW.WWWWW
..........r
WWWWW.WWWWW
WWWWW.WWWWW
g.B.....B.G
WWWWW.WWWWW
WWWWWZ....M
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["up", "up", "up", "up", "up", "right", "right", "right", "right", "right", "left", "left", "left", "left", "up", "up", "up", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["up", "up", "up", "up", "up", "up", "up", "up", "right", "right", "right", "right", "right", "up", "up", "down", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "left"],
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
          path: ["up", "up", "down", "right", "right", "up", "up", "down", "right", "right", "right", "up", "up", "down", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right"],
          goal: 3,
          type: 'Novice'
        },
        experienced2: {
          path: ["up", "up", "down", "right", "right", "up", "up", "down", "right", "right", "right", "up", "up", "down", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right"],
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
  stepsRemaining: 135,
  goal: {
    type: 'C',
    description: 'Find and obtain Treasure C'
  }
};
