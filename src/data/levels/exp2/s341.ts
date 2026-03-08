import { LevelConfig } from '../types';

export const s341: LevelConfig = {
  id: 's341',
  name: 's341',
  asciiMap:`
WWWWWeWWWgW
WWWWW.WWW.W
WWWWW.WWW.W
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
    2: {
      movements: {
        experienced1: {
          path: ["left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "down", "left", "left", "down", "down", "down"],
          goal: 2,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced3: {
          path: ["left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "down", "left", "left", "down", "down", "down"],
          goal: 2,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["right", "up", "up", "up"],
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
