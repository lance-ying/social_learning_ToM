import { LevelConfig } from '../types';

export const mod_s211: LevelConfig = {
  id: 'mod_s211',
  name: 'mod_s211',
  asciiMap: `
eWW.WW.WWWb
.WW.WW.WWW.
.WW.WW.WWW.
...........
WW.WWWWWWW.
..ZWWWWWWW.
WW.WWWWWWW.
WWMWWWWWWWW
WW.WWWWWWWW
.....WWWWWW
WWWWBWWWWWW
WWWWgWWWWWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: { 
          path: ["up", "up", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["up", "up", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["up", "up", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "down", "down"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["up", "up", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 135,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};