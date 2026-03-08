import { LevelConfig } from '../types';

export const old_s432: LevelConfig = {
  id: 'old_s432',
  name: 'old_s432',
  asciiMap: `
eWWWWbWWWWg
.WWWW.WWWWB
.WWWW.WWWW.
......WWWW.
.WWWWWWWWW.
...........
WWWW.WWWWW.
WWWWZWWWWWe
WWWW.WWWWWW
.....M....r
BWWWWRWWWWW
gWWWWgWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["up", "up", "left", "left", "left", "left", "up", "up", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 95,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};