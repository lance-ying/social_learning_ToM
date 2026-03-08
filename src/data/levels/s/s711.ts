import { LevelConfig } from '../types';

export const s711: LevelConfig = {
  id: 's711',
  name: 's711',
  asciiMap: `
WWWeeWWeeWW
WWW..WW..WW
e....O....b
WWWWW.WWWWW
WWWWW.WWWWW
gRB..Z....r
WWWWW.WWWWW
WWWWW.WWWWW
WWWWW.WWWWW
M..........
WWWWW.WWWW.
WWWWWgWWWWg
`.trim(),
  agentPaths: {
    1 : {
      movements: {
        experienced1: { 
          path: ["up", "up", "up", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["up", "up", "up", "left", "left", "up", "up", "down", "right", "up", "up", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["down", "down", "down", "down", "down", "down"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["down", "down", "down", "down", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    },
    2: {
      movements: {
        experienced1: { 
          path: ["right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["left", "up", "up", "down", "right", "right", "right", "up", "up", "down", "right", "up", "up", "down", "right", "right", "left", "left", "left", "left", "down", "down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["down", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["down", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 50,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
}; 