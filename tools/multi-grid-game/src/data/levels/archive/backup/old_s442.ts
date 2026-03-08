import { LevelConfig } from '../types';

export const old_s442: LevelConfig = {
  id: 'old_s442',
  name: 'old_s442',
  asciiMap: `
gWWWWgWWeWW
.WWWW.WW.WW
BWWWW.WW.WW
.....Z....e
WWWWW.WW.WW
WWWWW.WW.WW
WWWWW.WW.WW
r....MWW..e
WWWWW.WW.WW
...........
.WWWWWWW.WB
eWWWWWWWbWg
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["right", "right", "right", "down", "down", "down", "down", "down", "down", "down", "up", "right", "right", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["up", "up", "up"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["up", "up", "up", "up"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 60,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};