import { LevelConfig } from '../types';

export const wiz_05: LevelConfig = {
  id: 'wiz_05',
  name: 'Multi-agent wizard paths',
  asciiMap:`
bWWWWWWWWrW
.WW.....W.W
.WW.WWW.W.W
.WW.WWW.W.W
....M.....W
WWW.WWW.WWW
WWW.WWW.WWW
Z..........
WWWBWWWRWWW
WWW.WWW.WWW
WWW.WgW.WOW
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
          path: ["right", "left", "right", "left", "right", "left", "right", "right", "right", "up", "up", "down", "up", "down", "up", "down", "down", "up", "up", "down", "down", "right", "left", "up", "down", "up", "down", "right", "left", "up", "up", "up", "down", "down", "up", "down", "up", "up", "right", "left", "left", "right", "right", "left", "left", "right", "right", "left", "up", "up", "down", "up", "down", "up", "up", "right", "right", "left", "right", "left", "right", "left", "left", "right", "left", "down", "down", "up", "down", "up", "up", "down", "up", "down", "up", "right", "left", "right", "right", "right", "left", "right", "left", "left", "left", "down", "up", "down", "up", "right", "left", "right", "right", "left", "right", "right", "right", "left", "left", "left", "right", "left", "left", "down", "up", "right", "left", "down", "up", "down", "up", "down", "up", "down", "down", "down", "down", "down", "down", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "left", "right", "left", "right", "right", "left", "right", "right", "right", "right", "left", "left", "up", "up", "up", "left", "right", "left", "left", "right", "left"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: [],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["right", "right", "right", "right", "right", "left", "right", "left", "left", "up", "up", "down", "down", "up", "up", "up", "up", "down", "left", "right", "left", "left", "right", "left", "right", "right", "right", "right", "left", "left", "left", "right", "up", "down", "up", "up", "up", "right", "left", "right", "left", "right", "left", "down", "down", "down", "right", "left", "down", "up", "left", "right", "right", "left", "left", "left", "left", "up", "down", "right", "left", "right", "right", "left", "right", "left", "left", "right", "left", "right", "left", "up", "up", "down", "up", "down", "down", "right", "left", "up", "down", "up", "up", "down", "down", "right", "left", "up", "down", "up", "down", "right", "left", "right", "left", "right", "right", "left", "left", "up", "up", "up", "up", "down", "up", "down", "up", "down", "down", "up", "down", "down", "down", "up", "up", "up", "down", "up", "up", "down", "down", "down", "up", "down", "up", "up", "up", "down", "up", "down", "down", "down", "down", "right", "right", "right", "left", "right", "right", "right", "right", "right", "up", "up", "down", "down", "right", "left", "left", "left"],
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
