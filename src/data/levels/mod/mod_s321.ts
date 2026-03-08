import { LevelConfig } from '../types';

export const mod_s321: LevelConfig = {
  id: 'mod_s321',
  name: 'mod_s321',
  asciiMap: `
bWWWWWWWWW.
.WWWWWWWWW.
.WWWWWWWWW.
...........
WWWW.WWWWWW
WWWW.......
WWWW.WWWWWW
WWWW......e
WWWW.WWWWWW
M...Z.....e
BWWW.WWWWWW
gWWW.WWWWWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: { 
          path: ["up", "up", "up", "up", "up", "up", "left", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["up", "up", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "up", "up", "up", "up", "left", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["up", "up", "up", "up", "up", "up", "left", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["up", "up", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "up", "up", "up", "up", "left", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 135,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};