import { LevelConfig } from '../types';

export const mod_s544: LevelConfig = {
  id: 'mod_s544',
  name: 'mod_s544',
  asciiMap: `
WWWWbWeWWWW
WWWW.M.WWWW
...........
WWWWW.WWWWW
WWWWW.WWWWW
g.B........
WWWWW.WWWWW
WWWWW.WWWWW
WWWWW.WWWWW
Z..........
WWWWWWWWWWW
WWWWWWWWWWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: { 
          path: ["right", "right", "right", "right", "right", "up", "up", "up", "up", "up", "up", "up", "left", "up", "down", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 0,
          type: 'Expert'
        },
        experienced2: { 
          path: ["right", "right", "right", "right", "right", "up", "up", "up", "up", "up", "up", "up", "up", "right", "up", "left", "left", "up", "down", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 0,
          type: 'Novice'
        },
        experienced3: { 
          path: [],
          goal: 0,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["right", "right", "right", "right", "right", "up", "up", "up", "up", "up", "up", "up", "up", "right", "up", "left", "left", "up", "down", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 0,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 70,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};