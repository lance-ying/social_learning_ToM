import { LevelConfig } from '../types';

export const old_s341: LevelConfig = {
  id: 'old_s341',
  name: 'old_s341',
  asciiMap:`
WWWWWrWWWgW
eWWWW.WWWRW
.WWWW.WWW.W
........Z.W
.WWWWWW.WWW
bWWWWWW.WWW
WWWWWWW.WWW
WWWWWWW.WWW
..........M
WWWWW.WWW.W
WWWWWBWWW.W
WWWWWGWWWgW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: {
          path: ["left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "down", "left", "left", "down", "down", "down"],
          goal: 2,
          type: 'Expert'
        },
        experienced2: {
          path: ["left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "up", "up", "up"],
          goal: 1,
          type: 'Novice'
        },
        experienced3: {
          path: ["left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "down", "left", "left", "down", "down", "down"],
          goal: 2,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "up", "up", "up"],
          goal: 1,
          type: 'Novice_2'
        },
      }
    },
  },
  stepsRemaining: 130,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
