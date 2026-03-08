import { LevelConfig } from '../types';

export const old_s341: LevelConfig = {
  id: 'old_s341',
  name: 'old_s341',
  asciiMap: `
WWWWWrWWWgW
eWWWW.WWWRW
.WWWW.WWW.W
......Z...W
.WW.WWWWWWW
bWW.WWWWWWW
WWW.WWWWWWW
WWW.WWWWWWW
..........M
WWWWW.WWW.W
WWWWWBWWW.W
WWWWWgWWWgW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["left", "left", "left", "left", "left", "left", "down", "down", "up", "right", "right", "right", "down", "down", "down", "down", "down", "right", "right", "down", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["left", "up", "up", "down", "down", "right", "right", "right", "right", "up", "up", "up"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["left", "left", "left", "left", "left", "left", "up", "down", "down", "up", "right", "right", "right", "down", "down", "down", "down", "down", "right", "right", "down", "down", "down"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["left", "up", "up", "down", "down", "right", "right", "right", "right", "up", "up", "up"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 70,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};