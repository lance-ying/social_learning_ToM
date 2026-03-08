import { LevelConfig } from '../types';

export const s511: LevelConfig = {
  id: 's511',
  name: 's511',
  asciiMap: `
gWWWWgWWeWW
.WWWW.WW.WW
BWWWWRWW.WW
.....O....e
WWWWW.WW.WW
WWWWW.WW.WW
WWWWW.WW.WW
r....MWWZ.e
WWWWW.WW.WW
...........
.WWWWWWW.WB
eWWWWWWWbWg
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["down", "down", "down", "down", "up", "right", "right", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["right", "right", "left", "down", "down", "down", "down", "up", "right", "right", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["down", "down", "left", "left", "left", "up", "up", "left", "left", "left", "left", "left", "right", "right", "right", "right", "up", "up", "up", "up", "up", "up", "up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["down", "down", "left", "left", "left", "up", "up", "left", "left", "left", "left", "left", "right", "right", "right", "right", "up", "up", "up", "up", "up", "up", "up"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    },
    2: {
      movements: {
        experienced1: { 
          path: ["right", "right", "right", "down", "down", "down", "down", "down", "down", "down", "down", "up", "right", "right", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["right", "right", "right", "up", "up", "up", "down", "down", "down", "down", "down", "down", "right", "right", "left", "down", "down", "down", "down", "up", "right", "right", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["down", "down", "down", "down", "left", "left", "left", "left", "left", "right", "right", "right", "right", "up", "up", "up", "up", "up", "up", "up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["down", "down", "down", "down", "left", "left", "left", "left", "left", "right", "right", "right", "right", "up", "up", "up", "up", "up", "up", "up"],
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