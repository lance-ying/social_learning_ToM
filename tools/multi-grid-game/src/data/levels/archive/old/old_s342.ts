import { LevelConfig } from '../types';

export const old_s342: LevelConfig = {
  id: 'old_s342',
  name: 'old_s342',
  asciiMap: `
WWWWWrWWWgW
eWWWW.WWWRW
.WWWW.WWW.W
..Z.......W
.WW.WWW.WWW
bWW.WWW.WWW
WWW.WWW.WWW
WWW.WWW.WWW
..........M
WWWWWRWWW.W
WWWWWBWWW.W
WWWWWgWWWgW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "down", "down", "down", "down", "down", "right", "right", "down", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["right", "right", "right", "up", "up", "down", "down", "right", "right", "right", "right", "up", "up", "up"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["left", "left", "up", "down", "down", "up", "right", "right", "right", "right", "right", "up", "up", "down", "down", "left", "left", "down", "down", "down", "down", "down", "right", "right", "down", "down", "down", "down"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["right", "right", "right", "up", "up", "down", "down", "right", "right", "right", "right", "up", "up", "up"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 80,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};