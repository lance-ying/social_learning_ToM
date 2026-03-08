import { LevelConfig } from '../types';

export const wiz_01: LevelConfig = {
  id: 'wiz_01',
  name: 'Red wizard corridor',
  asciiMap:`
eWWrWWWeWWW
.WW.WW.WW.W
.WW.WW.WW.W
....M.....W
WWW.WWW.WWW
WWW.WWW.WWW
WWW.WWW.WWW
Z.........W
WWW.WWWRWWW
WWW.WWW.WWW
WWW.WWW.WgW
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
          path: ["right", "left", "right", "left", "right", "left", "right", "right", "right", "up", "up", "down", "up", "down", "up", "down", "down", "up", "up", "down", "down", "down", "up", "left", "left", "left", "right", "right", "left", "right", "left", "left", "right", "right", "left", "right", "left", "left", "right", "left", "right", "right", "right", "down", "down", "down", "up", "up", "up", "up", "down", "up", "down", "up", "up", "down", "down", "down", "down", "up", "down", "up", "up", "left", "left", "left", "right", "left", "right", "left", "right", "left", "right", "left", "right", "right", "left", "right", "right", "right", "left", "left", "left", "left", "right", "left", "right", "left", "right", "right", "left", "right", "right", "down", "down", "down", "up", "down", "up", "up", "down", "up", "up", "down", "up", "right", "left", "down", "up", "up", "up", "up", "up", "down", "down", "down", "up", "down", "down", "right", "right", "left", "left", "up", "down", "down", "up", "up", "up", "up", "up", "right", "right", "left", "right", "right", "right", "left", "up", "up", "down", "up", "down", "down", "left", "right", "right", "right", "right", "up"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: [],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["right", "right", "right", "left", "left", "left", "right", "left", "right", "left", "right", "right", "left", "left", "right", "left", "right", "right", "right", "left", "left", "right", "left", "right", "right", "right", "right", "left", "left", "left", "right", "up", "down", "up", "up", "up", "down", "up", "down", "up", "down", "up", "up", "right", "right", "right", "up", "up", "down", "down", "right", "right", "right", "up", "down", "up", "down", "up", "down", "left", "right", "up", "down", "up", "up", "down", "up", "down", "up", "down", "up", "down", "up", "down", "down", "left", "left", "right", "left", "left", "right", "down", "down", "down", "down", "up", "down", "right", "right", "left", "right", "left", "right", "left", "right", "left", "left", "up", "up", "up", "up", "down", "down", "up", "up", "right", "left", "left", "left", "right", "right", "left", "right", "down", "up", "right", "right", "up", "up", "down", "down", "left", "left", "left", "up", "up", "down", "down", "up", "down", "left", "right", "right", "left", "left", "right", "right", "right", "right", "left", "right", "left", "left", "down", "down", "down", "down", "up", "down", "up"],
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
