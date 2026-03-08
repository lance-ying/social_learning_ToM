import { LevelConfig } from '../types';

export const old_s511: LevelConfig = {
  id: 'old_s511',
  name: 'old_s511',
  asciiMap: `
eWWeWWeWWWb
.WW.WW.WWW.
.WW.WW.WWW.
...........
WW.WWWWWWW.
r.ZWWWWWWW.
WW.WWWWWWWe
WWMWWWWWWWW
WW.WWWWWWWW
.........Rg
BWWWBWWWWWW
gWWWgWWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["up", "up", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 80,
  goal: {
    type: 'C',
    description: 'Find and obtain Treasure C'
  }
};