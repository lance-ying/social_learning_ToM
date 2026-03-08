import { LevelConfig } from '../types';

export const wiz_04: LevelConfig = {
  id: 'wiz_04',
  name: 'Central wizard chamber',
  asciiMap:`
WWWWWWWWWWW
WZ........W
WWW.WWWWWWW
WWW.WWeWWWW
WWW.WW.WWWW
WWW....WWWW
WWWWWWRWWWW
WWWWWWWWWWW
WM........W
WWW.WWWWWWW
WWW.WgWWWWW
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
          path: ["right", "left", "right", "left", "right", "left", "right", "right", "right", "left", "down", "down", "up", "down", "up", "down", "down", "up", "up", "down", "down", "up", "up", "down", "up", "up", "left", "right", "down", "down", "up", "up", "down", "down", "up", "down", "up", "up", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "right", "left", "right", "left", "left", "right", "right", "left", "right", "left", "right", "left", "left", "right", "left", "left", "right", "left", "right", "left", "left", "down", "up", "down", "up", "right", "left", "right", "right", "right", "right", "left", "right", "left", "left", "left", "left", "down", "up", "down", "down", "up", "down", "down", "up", "down", "down", "right", "right", "left", "left", "right", "left", "up", "up", "up", "down", "up", "up", "down", "up", "down", "up", "down", "up", "left", "right", "down", "down", "down", "down", "right", "left", "up", "up", "down", "up", "up", "up", "right", "right", "left", "left", "left", "right", "down", "down", "down", "down", "up", "up", "up", "up", "down", "down", "down", "down", "up", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: [],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["right", "right", "left", "right", "down", "up", "right", "left", "down", "up", "right", "right", "right", "left", "left", "left", "down", "down", "down", "down", "right", "left", "right", "left", "right", "right", "right", "left", "left", "left", "right", "right", "left", "right", "left", "left", "up", "down", "up", "down", "up", "down", "up", "up", "down", "down", "right", "left", "up", "up", "down", "down", "right", "left", "right", "left", "up", "up", "down", "down", "up", "down", "right", "left", "right", "left", "up", "down", "up", "down", "up", "up", "up", "down", "up", "down", "down", "down", "up", "up", "down", "up", "up", "down", "down", "down", "up", "up", "down", "up", "down", "down", "up", "down", "up", "down", "right", "left", "up", "up", "up", "up", "down", "down", "up", "up", "right", "left", "left", "right", "left", "right", "right", "left", "down", "down", "up", "up", "left", "right", "right", "right", "right", "left", "left", "left", "left", "left", "right", "right", "left", "right", "left", "right", "left", "right", "left", "right", "left", "right", "left", "left", "right", "right", "left", "right", "down", "down", "up", "up"],
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
