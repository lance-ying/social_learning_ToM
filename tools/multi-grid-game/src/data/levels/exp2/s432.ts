import { LevelConfig } from '../types';

export const s432: LevelConfig = {
  id: 's432',
  name: 's432',
  asciiMap:`
eWeWbWWWWgW
.W.W.WWWWBW
.W.W.WWWW.W
.....WWWW.W
.WWWWWWWW.W
....M......
WWW.WWWWWW.
WWW.WWWWWWe
WWW.WWWWWWW
.....Z...rW
BWWWRWWWWWW
GWWWgWWWWWW

`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["left", "left", "up", "up", "up", "up", "left", "left", "left", "up", "up", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right", "right", "right", "left", "left", "left", "left", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced3: {
          path: ["left", "left", "up", "up", "up", "up", "left", "left", "left", "up", "up", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "up", "up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["right", "right", "right", "right", "left", "left", "left", "left", "down", "down"],
          goal: 3,
          type: 'Novice_2'
        },
      }
    },
  },
  stepsRemaining: 125,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
