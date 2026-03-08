import { LevelConfig } from '../types';

export const old_s331: LevelConfig = {
  id: 'old_s331',
  name: 'old_s331',
  asciiMap:`
WWWWeWWWWWg
WWWW.WWWWWB
b..Z.......
WWW.WWWWWWW
WWW.WWWWWWW
WWW.WWWWWWW
WWW.WWWWWWW
..........r
RWW.WWW.WWW
.WW.WWW.WWW
.WW.WWW..BG
gWWMWWWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: {
          path: ["left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down"],
          goal: 3,
          type: 'Novice'
        },
        experienced3: {
          path: ["left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down"],
          goal: 3,
          type: 'Novice_2'
        },
      }
    },
  },
  stepsRemaining: 95,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
