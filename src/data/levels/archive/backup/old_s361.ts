import { LevelConfig } from '../types';

export const old_s361: LevelConfig = {
  id: 'old_s361',
  name: 'old_s361',
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
gWWWgWW.B.g
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["up", "up", "right", "right", "right", "right", "right", "right", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "down", "down", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["down", "down", "down", "down", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["down", "down", "down", "down", "down", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 50,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};