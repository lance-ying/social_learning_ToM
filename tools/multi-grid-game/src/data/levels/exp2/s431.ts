import { LevelConfig } from '../types';

export const s431: LevelConfig = {
  id: 's431',
  name: 's431',
  asciiMap:`
eWeWbWWWWgW
.W.W.WWWWBW
.W.W.WWWW.W
.....WWWW.W
.WWWWWWWW.W
....Z......
WWW.WWWWWW.
WWW.WWWWWWe
WWW.WWWWWWW
.....M...rW
BWWWRWWWWWW
GWWWgWWWWWW\
    
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["left", "left", "left", "left", "up", "up", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "left", "left", "left", "left", "up"],
          goal: 2,
          type: 'Expert'
        },
        experienced3: {
          path: ["left", "left", "left", "left", "up", "up", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "up", "up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "left", "left", "left", "left", "up"],
          goal: 2,
          type: 'Novice_2'
        },
      }
    },
  },
  stepsRemaining: 140,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
