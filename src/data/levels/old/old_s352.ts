import { LevelConfig } from '../types';

export const old_s352: LevelConfig = {
  id: 'old_s352',
  name: 'old_s352',
  asciiMap: `
WWWWWeWeWWb
WWWWW.W.WW.
g..........
WWWWW.WWWWW
WWWWWZ.....
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
          path: ["up", "up", "right", "right", "right", "right", "right", "up", "down", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["up", "up", "left", "left", "left", "left", "left"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["up", "up", "left", "left", "left", "left", "left"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 65,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};