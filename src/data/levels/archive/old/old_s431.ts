import { LevelConfig } from '../types';

export const old_s431: LevelConfig = {
  id: 'old_s431',
  name: 'old_s431',
  asciiMap: `
eWWWWbWWWWg
.WWWW.WWWWB
.WWWW.WWWW.
......WWWW.
.WWWWWWWWW.
Z..........
WWWW.WWWWW.
WWWW.WWWWWe
WWWW.WWWWWW
.....M....r
BWWWWRWWWWW
gWWWWgWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["up", "up", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["right", "right", "right", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["right", "right", "right", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 90,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};