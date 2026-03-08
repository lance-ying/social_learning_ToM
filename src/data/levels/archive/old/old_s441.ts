import { LevelConfig } from '../types';

export const old_s441: LevelConfig = {
  id: 'old_s441',
  name: 'old_s441',
  asciiMap: `
gWWWWgWWeWW
.WWWW.WW.WW
BWWWW.WW.WW
.....M....e
WWWWW.WW.WW
WWWWW.WW.WW
WWWWW.WW.WW
r....ZWW..e
WWWWW.WW.WW
...........
.WWWWWWW.WB
eWWWWWWWbWg
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["down", "down", "right", "right", "right", "down", "down", "up", "up", "up", "up", "up", "up", "up", "left", "left", "left", "left", "left", "left", "left", "left", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["up", "up", "up", "up", "up", "up", "up"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["up", "up", "up", "up", "up", "up", "up", "up"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 65,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};