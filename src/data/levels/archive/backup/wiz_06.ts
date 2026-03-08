import { LevelConfig } from '../types';

export const wiz_06: LevelConfig = {
  id: 'wiz_06',
  name: 'Wizard gauntlet',
  asciiMap:`
eWWbWWrWWWW
.WW.WW.WWWW
.WW.WW.WWWW
.......WWWW
WWWWWWWWWWW
WWWWWWWWWWW
WM........W
WWW.WWWWWWW
WWWBWWWRWWW
WWW.WWW.WWW
WWZ.WgW.WWW
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
          path: ["right", "up", "down", "up", "down", "up", "down", "left", "right", "up", "down", "left", "right", "left", "right", "left", "right", "up", "down", "left", "right", "up", "down", "left", "right", "up", "down", "left", "right", "left", "right", "up", "down", "left", "right", "left", "right", "up", "down", "up", "down", "left", "right", "up", "down", "left", "right", "up", "down", "up", "down", "up", "down", "up", "down", "left", "right", "up", "down", "up", "down", "up", "down", "left", "right", "up", "down", "up", "down", "up", "down", "up", "down", "up", "down", "left", "right", "left", "right", "left", "right", "left", "right", "up", "down", "up", "down", "up", "down", "left", "right", "left", "right", "up", "down", "left", "right", "left", "right", "up", "down", "up", "down", "up", "down", "left", "right", "up", "down", "up", "down", "up", "down", "up", "down", "left", "right", "left", "right", "left", "right", "up", "down", "up", "down", "up", "down", "up", "down", "up", "down", "left", "right", "up", "down", "left", "right", "up", "down", "up", "down", "up", "down", "left", "right", "up", "down", "up", "down", "up"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: [],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["right", "left", "right", "left", "right", "up", "down", "up", "down", "up", "down", "left", "right", "up", "down", "up", "down", "left", "right", "left", "right", "left", "right", "left", "right", "left", "right", "up", "down", "left", "right", "up", "down", "up", "down", "up", "down", "up", "down", "up", "down", "up", "down", "left", "right", "left", "right", "up", "down", "left", "right", "left", "right", "left", "right", "up", "down", "left", "right", "up", "down", "left", "right", "left", "right", "up", "down", "up", "down", "up", "down", "up", "down", "up", "down", "left", "right", "up", "down", "left", "right", "up", "down", "left", "right", "up", "down", "left", "right", "left", "right", "up", "down", "up", "down", "left", "right", "up", "down", "up", "down", "up", "down", "up", "down", "up", "down", "left", "right", "left", "right", "left", "right", "up", "down", "left", "right", "up", "down", "left", "right", "up", "down", "up", "down", "up", "down", "up", "down", "left", "right", "left", "right", "left", "right", "left", "right", "left", "right", "left", "right", "up", "down", "left", "right", "left", "right", "left", "right", "up"],
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
