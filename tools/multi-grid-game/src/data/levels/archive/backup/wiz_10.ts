import { LevelConfig } from '../types';

export const wiz_10: LevelConfig = {
  id: 'wiz_10',
  name: 'Complex multi-path',
  asciiMap:`
bWWeWWrWWWW
.WW.WW.WWWW
.WW.WW.WWWW
....WW....W
WWWWWWWWWWW
WZ........W
WWW.WWWWWWW
WWM.WWOWWWW
WWWBWWWRWWW
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
          path: ["right", "left", "right", "left", "right", "left", "right", "right", "right", "left", "down", "down", "up", "down", "up", "down", "left", "right", "up", "down", "left", "right", "up", "down", "up", "up", "left", "right", "down", "down", "up", "up", "down", "down", "up", "down", "up", "up", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "right", "left", "right", "left", "left", "right", "right", "left", "right", "left", "right", "left", "left", "right", "left", "left", "right", "left", "right", "left", "left", "down", "up", "down", "up", "right", "left", "right", "right", "right", "right", "left", "right", "left", "left", "left", "left", "down", "up", "down", "down", "up", "down", "left", "right", "left", "right", "left", "right", "up", "up", "left", "left", "right", "left", "right", "right", "down", "up", "down", "up", "down", "up", "down", "up", "left", "right", "down", "down", "left", "right", "left", "right", "up", "up", "left", "left", "right", "left", "right", "left", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right", "left", "right", "left"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: [],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["right", "right", "left", "right", "down", "up", "right", "left", "down", "up", "right", "right", "right", "left", "left", "left", "down", "down", "left", "right", "left", "right", "left", "right", "left", "right", "left", "right", "up", "up", "right", "right", "right", "left", "right", "right", "left", "right", "left", "right", "right", "left", "right", "left", "left", "right", "right", "right", "left", "left", "left", "right", "right", "right", "left", "right", "left", "left", "left", "right", "right", "left", "right", "right", "left", "right", "left", "left", "right", "left", "right", "left", "left", "left", "right", "left", "right", "right", "right", "left", "left", "right", "left", "left", "right", "right", "right", "left", "left", "right", "left", "right", "right", "left", "right", "left", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right", "left", "left", "left", "right", "down", "down", "left", "right", "up", "up", "down", "down", "up", "up", "left", "right", "right", "right", "right", "left", "left", "left", "left", "left", "right", "right", "left", "right", "left", "right", "left", "right", "left", "right", "left", "right", "left", "left", "right", "right", "left", "right", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        },
      }
    },
    2: {
      movements: {
        experienced1: { 
          path: [],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: [],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: [],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: [],
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
