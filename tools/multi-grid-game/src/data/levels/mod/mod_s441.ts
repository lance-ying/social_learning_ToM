import { LevelConfig } from '../types';

export const mod_s441: LevelConfig = {
  id: 'mod_s441',
  name: 'mod_s441',
  asciiMap: `
gWWWWWWWeWW
.WWWWWWW.WW
BWWWWWWW.WW
.....M....e
WWWWW.WW.WW
WWWWW.WW.WW
WWWWW.WW.WW
.....ZWW..e
WWWWW.WW.WW
...........
.WWWWWWW.WW
eWWWWWWWbWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: { 
          path: ["down", "down", "right", "right", "right", "down", "down", "up", "up", "up", "up", "up", "up", "up", "left", "left", "left", "left", "left", "left", "left", "left", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["down", "down", "right", "right", "right", "up", "up", "right", "right", "left", "down", "down", "down", "down", "up", "up", "up", "up", "up", "up", "up", "left", "left", "left", "left", "left", "left", "left", "left", "up", "up", "up"],
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
  stepsRemaining: 120,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};