import { LevelConfig } from '../types';

export const wiz_07: LevelConfig = {
  id: 'wiz_07',
  name: 'Separated chambers',
  asciiMap:`
rWWWWWWWWWW
.WWWWWWWWWW
.WW.......W
.WW.WWWWW.W
.WW.WWbWW.W
....WWW...W
WWWWWWWWWWW
ZM........W
WWWRWWWBWWW
WWW.WWW.WWW
WWW.WgW.WWW
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
          path: ["right", "left", "right", "left", "right", "left", "right", "right", "right", "left", "left", "right", "left", "right", "left", "right", "right", "left", "left", "right", "right", "left", "left", "right", "left", "left", "right", "right", "left", "right", "left", "left", "right", "right", "left", "right", "left", "left", "right", "left", "right", "right", "right", "left", "right", "right", "right", "left", "left", "left", "right", "left", "right", "left", "left", "right", "right", "left", "right", "left", "right", "left", "left", "right", "left", "left", "right", "left", "right", "left", "right", "left", "right", "left", "right", "right", "left", "right", "right", "right", "left", "right", "left", "left", "left", "left", "right", "left", "right", "right", "left", "right", "right", "left", "right", "right", "right", "right", "left", "left", "right", "left", "left", "left", "left", "right", "left", "left", "right", "left", "right", "left", "right", "left", "right", "right", "left", "right", "right", "right", "right", "left", "left", "left", "right", "left", "left", "left", "right", "left", "right", "right", "right", "left", "right", "right", "right", "left", "left", "left", "left", "left", "right", "right", "right", "left", "right", "left", "left", "left"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: [],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["right", "right", "right", "right", "left", "left", "left", "left", "right", "left", "right", "right", "left", "left", "right", "left", "right", "right", "right", "right", "left", "right", "left", "right", "right", "right", "right", "left", "left", "right", "right", "left", "right", "left", "left", "left", "right", "left", "right", "left", "right", "left", "left", "right", "right", "right", "left", "left", "left", "right", "right", "right", "left", "right", "left", "left", "left", "right", "right", "left", "right", "right", "left", "right", "left", "left", "right", "left", "right", "left", "left", "left", "right", "left", "right", "right", "right", "left", "left", "right", "left", "left", "right", "right", "right", "left", "left", "right", "left", "right", "right", "left", "right", "left", "right", "right", "left", "left", "left", "left", "left", "left", "right", "left", "right", "left", "right", "right", "left", "right", "right", "right", "left", "left", "left", "right", "left", "left", "right", "right", "right", "left", "right", "left", "left", "left", "right", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "left", "right", "left", "right", "left", "left", "right", "right", "left", "left", "right", "left", "left"],
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
