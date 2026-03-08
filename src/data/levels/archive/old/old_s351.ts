import { LevelConfig } from '../types';

export const old_s351: LevelConfig = {
  id: 'old_s351',
  name: 'old_s351',
  asciiMap: `
WWWWWeWeWWb
WWWWW.W.WW.
g...Z......
WWWWW.WWWWW
WWWWW......
WWWWW.WWWWW
WWWWW.WWWWW
g.B..M..B.g
WWWWW.WWWWW
WWWWW.WWWWW
...........
WWWWWWWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["right", "right", "right", "right", "right", "right", "up", "up", "down", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["left", "left", "left", "left"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["left", "left", "left", "left"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 55,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};