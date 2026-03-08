import { LevelConfig } from '../types';

export const wiz_02: LevelConfig = {
  id: 'wiz_02',
  name: 'Blue wizard maze',
  asciiMap:`
bWWWWWWWWWW
.WWWWWWWWWW
.WW.......W
.WW.WWWWW.W
.WW.WWWWW.W
.....M.Z..W
WWWWWWW.WWW
WWWWWWW.WWW
WWWWWWW.WWW
WWWWWWWBgWW
WWWWWWW.WWW
WWWWWWWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["left", "left", "left", "left", "left", "left", "left", "up", "up", "up", "up", "up", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["left", "left", "left", "left", "right", "left", "up", "up", "up", "right", "right", "left", "left", "right", "left", "right", "left", "right", "right", "left", "left", "right", "right", "left", "left", "right", "left", "down", "down", "down", "up", "down", "right", "left", "up", "up", "down", "up", "down", "up", "up", "right", "left", "right", "right", "right", "left", "right", "right", "right", "left", "left", "left", "right", "left", "right", "left", "left", "right", "right", "left", "right", "left", "right", "left", "left", "right", "left", "left", "right", "left", "right", "left", "down", "up", "down", "up", "down", "down", "up", "down", "down", "right", "left", "left", "left", "left", "up", "up", "up", "up", "up", "down", "up", "down", "down", "up", "down", "down", "down", "down", "up", "up", "down", "up", "up", "up", "up", "down", "up", "down", "up", "down", "up", "down", "up", "down", "down", "down", "up", "down", "down", "down", "right", "left", "up", "up", "down", "up", "up", "up", "up", "down", "up", "down", "down", "up", "down", "down", "down", "up", "up", "up", "up", "down", "down", "down", "down", "up", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["left", "left", "left", "left", "left", "left", "left", "up", "up", "up", "up", "up", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "right"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["left", "left", "right", "right", "down", "up", "right", "left", "down", "up", "right", "right", "left", "left", "down", "up", "down", "down", "down", "up", "down", "up", "down", "up", "down", "up", "down", "up", "up", "up", "right", "right", "left", "left", "left", "right", "down", "down", "up", "down", "down", "up", "down", "up", "up", "down", "down", "up", "up", "up", "down", "down", "down", "up", "up", "down", "up", "up", "down", "down", "down", "up", "down", "up", "up", "down", "up", "up", "left", "left", "right", "left", "left", "left", "left", "left", "right", "right", "left", "left", "left", "right", "left", "up", "down", "right", "right", "left", "left", "right", "left", "right", "right", "left", "right", "left", "right", "right", "up", "up", "up", "down", "up", "down", "down", "up", "up", "down", "down", "down", "up", "down", "left", "right", "up", "up", "up", "right", "left", "down", "down", "down", "right", "right", "right", "left", "left", "left", "left", "left", "right", "right", "left", "right", "left", "right", "left", "right", "left", "right", "left", "right", "left", "left", "left", "right", "right", "right", "up", "down"],
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
