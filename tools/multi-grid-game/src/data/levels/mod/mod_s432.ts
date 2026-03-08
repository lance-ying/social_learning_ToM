import { LevelConfig } from '../types';

export const mod_s432: LevelConfig = {
  id: 'mod_s432',
  name: 'mod_s432',
  asciiMap: `
eWWWWbWWWWW
.WWWW.WWWWW
.WWWW.WWWWW
......WWWWW
.WWWWWWWWWW
...........
WWWW.WWWWW.
WWWWZWWWWWe
WWWW.WWWWWW
R....M....r
BWWWWWWWWWW
gWWWWWWWWWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: { 
          path: ["up", "up", "left", "left", "left", "left", "up", "up", "right", "right", "right", "right", "right", "up", "up", "down", "down", "left", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["up", "up", "left", "left", "left", "left", "up", "up", "up", "up", "up", "down", "down", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: [],
          goal: 0,
          type: 'Expert_2'
        },
        experienced4: { 
          path: [],
          goal: 0,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 190,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};