import { LevelConfig } from '../types';

export const old_s542: LevelConfig = {
  id: 'old_s542',
  name: 'old_s542',
  asciiMap: `
WWWbeWWeeWW
WWW..WW..WW
.....Z.....
WWWWW...WWW
WWWWW.WWWWW
gRB.......r
WWWWW.WWWWW
WWWWW.WWWWW
WWWWW.WWWWW
M..........
WWWWW.WWWW.
WWWWWgWWWWg
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["left", "left", "up", "down", "right", "right", "down", "down", "down", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["down", "down", "down", "down", "down", "down", "down", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["right", "right", "up", "down", "left", "left", "left", "left", "up", "down", "right", "right", "down", "down", "down", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["down", "down", "down", "down", "down", "down", "down", "down", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 80,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};