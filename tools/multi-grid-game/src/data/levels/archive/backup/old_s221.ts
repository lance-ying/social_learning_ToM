import { LevelConfig } from '../types';

export const old_s221: LevelConfig = {
  id: 'old_s221',
  name: 'old_s221',
  asciiMap: `
bWWWWWWWWW.
.WWWWWWWWW.
.WWWWWWWWW.
...........
WWWW.WWWWWW
WWWW.......
WWWW.WWWWWW
gWWW.......
BWWW.WWWWWW
M...Z.....e
BWWW.WWWWWW
gWWW......g
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["up", "up", "up", "up", "up", "up", "left", "left", "left", "left", "up", "up", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["down", "down", "right", "right", "right", "right", "right", "right"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["down", "down", "right", "right", "right", "right", "right"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 85,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};