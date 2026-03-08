import { LevelConfig } from '../types';

export const old_s371: LevelConfig = {
  id: 'old_s371',
  name: 'old_s371',
  asciiMap: `
WWWWWWWWWgW
eWWWWWWWWBW
.WWWWWWWW.W
M.........W
.WW.WWWWWWW
bWW.WWWWWWW
WWW.WWWWWWW
WWW.WWWWWWW
..........Z
WWWWW.WWW.W
WWWWW.WWW.W
WWWWWgWWWgW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["left", "left", "left", "left", "left", "left", "left", "up", "up", "up", "up", "up", "left", "left", "left", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up"],
          goal: 0,
          type: 'Expert'
        },
        experienced2: { 
          path: ["left", "left", "left", "left", "left", "down", "down", "down"],
          goal: 0,
          type: 'Novice'
        },
        experienced3: { 
          path: [],
          goal: 0,
          type: 'Expert_2'
        },
        experienced4: { 
          path: [],
          goal: 0,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 45,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};