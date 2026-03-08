import { LevelConfig } from '../types';

export const s611: LevelConfig = {
  id: 's611',
  name: 's611',
  asciiMap: `
eWWWgWW.WWg
.WWW.WW.WWB
.WWW.WW.WWR
...........
WWWWZWWWWWW
b.........e
WWWW.WWWWWW
..........e
.WWW.WWWWWW
M...O.....e
BWWW.WWWWWW
gWWW......r
`.trim(),
  agentPaths: {
    1   : {
      movements: {
        experienced1: { 
          path: ["down", "left", "left", "left", "left", "right", "right", "right", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["up", "left", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "down", "down", "left", "left", "left", "left", "right", "right", "right", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["up", "up", "up", "up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["up", "up", "up", "up"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    },
    2: {
      movements: {
        experienced1: { 
          path: ["up", "up", "up", "up", "left", "left", "left", "left", "right", "right", "right", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "up", "up", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "up", "up", "left", "left", "left", "left", "right", "right", "right", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["up", "up", "up", "up", "up", "up", "up", "up", "up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["up", "up", "up", "up", "up", "up", "up", "up", "up"],
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