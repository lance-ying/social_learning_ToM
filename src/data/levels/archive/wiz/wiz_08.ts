import { LevelConfig } from '../types';

export const wiz_08: LevelConfig = {
  id: 'wiz_08',
  name: 'Corner wizard puzzle',
  asciiMap:`
eWWWWWWWWWb
.WW.....W.W
.WW.WWW.W.W
.WW.WWW.W.W
.WW.......W
.WWWWWWWWWW
.M.......ZW
WWWWWWWWWWW
WWWWWWWWWWW
WWWWWWWBgWW
WWWWWWr.WWW
WWWWWWWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: [],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["left", "left", "left", "left", "left", "left", "left", "right", "right", "left", "left", "right", "left", "right", "left", "right", "right", "left", "left", "right", "right", "left", "left", "right", "left", "left", "right", "right", "left", "right", "left", "left", "left", "right", "left", "right", "left", "up", "down", "up", "down", "right", "right", "left", "right", "right", "right", "left", "left", "left", "right", "left", "right", "left", "left", "right", "right", "left", "right", "left", "right", "left", "left", "right", "left", "up", "down", "up", "down", "up", "up", "up", "up", "up", "up", "down", "up", "down", "down", "down", "up", "down", "up", "up", "up", "down", "up", "down", "up", "down", "up", "down", "down", "up", "down", "down", "down", "down", "up", "up", "down", "up", "up", "up", "up", "down", "up", "down", "up", "down", "up", "down", "up", "down", "down", "down", "up", "down", "down", "down", "down", "up", "up", "up", "down", "up", "up", "up", "up", "down", "up", "down", "down", "up", "down", "down", "down", "up", "up", "up", "up", "down", "down", "down", "down", "up", "down", "up", "up", "up"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: [],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["left", "right", "left", "right", "left", "left", "left", "left", "left", "left", "right", "right", "left", "left", "left", "left", "right", "right", "right", "right", "left", "right", "left", "right", "right", "right", "right", "left", "left", "right", "right", "left", "right", "left", "left", "left", "right", "left", "right", "left", "right", "left", "left", "right", "right", "right", "left", "left", "left", "right", "right", "right", "left", "right", "left", "left", "left", "right", "right", "left", "right", "right", "left", "right", "left", "left", "right", "left", "right", "left", "left", "left", "right", "left", "right", "right", "right", "left", "left", "right", "left", "left", "right", "right", "right", "left", "left", "right", "left", "right", "right", "left", "right", "left", "right", "right", "left", "left", "left", "left", "left", "left", "right", "left", "left", "up", "down", "right", "left", "right", "right", "right", "left", "left", "left", "right", "left", "up", "down", "right", "right", "left", "right", "left", "left", "up", "down", "up", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "left", "right", "left", "left", "left", "right", "right", "right", "left", "right", "left", "left"],
          goal: 2,
          type: 'Novice_2'
        },
      }
    },
  },
  stepsRemaining: 50,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
