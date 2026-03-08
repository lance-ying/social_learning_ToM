import { LevelConfig } from '../types';

export const wiz_09: LevelConfig = {
  id: 'wiz_09',
  name: 'Tight maze navigation',
  asciiMap:`
WWWWWWWWWWW
WrW.......W
W.W.WWWWW.W
W.W.WWbWW.W
W...WW.WW.W
WWW.WW.WW.W
WWW.WW....W
WM..WW.Z..W
WWW.WWWWWWW
WWWRWWWBWWW
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
          path: ["up", "down", "right", "right", "up", "left", "left", "down", "up", "down", "right", "right", "left", "up", "down", "left", "up", "left", "up", "down", "down", "up", "right", "down", "right", "left", "right", "left", "right", "up", "left", "down", "up", "left", "right", "down", "up", "right", "right", "up", "up", "up", "down", "up", "up", "up", "left", "left", "right", "right", "left", "left", "right", "right", "left", "left", "left", "left", "right", "left", "right", "left", "left", "right", "right", "left", "right", "left", "right", "left", "left", "right", "left", "left", "right", "left", "right", "left", "down", "up", "down", "up", "down", "down", "up", "down", "down", "left", "left", "right", "left", "up", "up", "up", "down", "up", "down", "down", "up", "down", "down", "up", "down", "right", "right", "down", "up", "left", "right", "up", "up", "down", "up", "down", "up", "up", "down", "up", "down", "up", "down", "up", "right", "right", "left", "right", "right", "right", "right", "left", "left", "left", "right", "left", "left", "left", "down", "up", "down", "down", "down", "up", "down", "down", "down", "up", "up", "up", "up", "up"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: [],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["left", "right", "left", "right", "up", "down", "right", "up", "down", "up", "right", "down", "left", "up", "down", "up", "down", "right", "left", "left", "up", "left", "down", "right", "up", "left", "right", "right", "right", "down", "left", "right", "up", "up", "down", "left", "left", "down", "left", "right", "up", "right", "left", "down", "up", "right", "down", "left", "right", "right", "up", "down", "left", "left", "up", "down", "up", "left", "right", "right", "left", "down", "right", "left", "up", "down", "left", "right", "up", "left", "down", "up", "down", "up", "up", "down", "up", "down", "right", "down", "right", "left", "right", "left", "up", "left", "down", "right", "right", "up", "right", "down", "left", "up", "right", "up", "down", "down", "left", "up", "right", "left", "right", "up", "down", "left", "down", "left", "up", "down", "left", "right", "up", "down", "right", "up", "down", "up", "down", "right", "left", "right", "up", "down", "left", "left", "right", "left", "up", "down", "up", "right", "left", "right", "down", "left", "right", "left", "left", "up", "up", "up", "down", "up", "down", "down", "down", "right", "left", "right"],
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
