import { LevelConfig } from '../types';

export const s312: LevelConfig = {
  id: 's312',
  name: 's312',
  asciiMap: `
eWW.WW.WWWb
.WW.WW.WWW.
.WW.WW.WWW.
.....Z....O
WW.WWW.WWW.
r..WWW.WWW.
WW.WWW.WWWe
WW.WWWWWWWW
WW.WWWWWWWW
..M......Rg
BWWWBWWWWWW
gWWWgWWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["right", "right", "right", "right", "right", "down", "down", "down", "up", "up", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["left", "left", "left", "down", "down", "left", "left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["left", "left", "left", "down", "down", "left", "left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    },
    2: {
      movements: {
        experienced1: { 
          path: ["up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["down", "down", "down", "up", "up", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "left", "left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "left", "left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"],
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