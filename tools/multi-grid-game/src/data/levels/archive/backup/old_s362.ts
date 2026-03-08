import { LevelConfig } from '../types';

export const old_s362: LevelConfig = {
  id: 'old_s362',
  name: 'old_s362',
  asciiMap: `
eWWWWWWWWWb
.WWWWWWWWW.
.WWWWWWWWW.
....M......
WWWW.WWWWWW
WWWW......r
WWWW.WWWWWW
e...ZWWWWWW
WWWW.WWWWWW
...........
RWWW.WW.WWW
gWWWgWW.B.g
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["up", "up", "up", "up", "right", "right", "right", "right", "right", "right", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "down", "down", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["up", "up", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["up", "up", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 20,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};