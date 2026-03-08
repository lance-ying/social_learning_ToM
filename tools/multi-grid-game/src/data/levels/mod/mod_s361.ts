import { LevelConfig } from '../types';

export const mod_s361: LevelConfig = {
  id: 'mod_s361',
  name: 'mod_s361',
  asciiMap: `
eWWWWWWWWWb
.WWWWWWWWW.
.WWWWWWWWW.
...........
WWWW.WWWWWW
WWWWZ.....r
WWWW.WWWWWW
e...MWWWWWW
WWWW.WWWWWW
...........
RWWW.WW.WWW
gWWW.WW.WWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: { 
          path: ["right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "down", "down", "left", "left", "left", "left", "right", "right", "right", "down", "down", "left", "left", "left", "left", "down", "down"],
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
   
  stepsRemaining: 100,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};