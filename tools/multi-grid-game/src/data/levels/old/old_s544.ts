import { LevelConfig } from '../types';

export const old_s544: LevelConfig = {
  id: 'old_s544',
  name: 'old_s544',
  asciiMap: `
WWWWbWeWWWW
WWWW.M.WWWW
...........
WWWWW.WWWWW
WWWWW.WWWWW
g.B.......r
WWWWW.WWWWW
WWWWW.WWWWW
WWWWW.WWWWW
Z..........
WWWWW.WWWWR
WWWWWgWWWWg
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["right", "right", "right", "right", "right", "up", "up", "up", "up", "up", "up", "up", "left", "up", "down", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 0,
          type: 'Expert'
        },
        experienced2: { 
          path: ["right", "right", "right", "right", "right", "up", "up", "up", "up", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 0,
          type: 'Novice'
        },
        experienced3: { 
          path: [],
          goal: 0,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["right", "right", "right", "right", "right", "up", "up", "up", "up", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 0,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 40,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};