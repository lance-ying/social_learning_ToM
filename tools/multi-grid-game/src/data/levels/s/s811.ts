import { LevelConfig } from '../types';

export const s811: LevelConfig = {
  id: 's811',
  name: 's811',
  asciiMap: `
WeWeWgWeWbW
W.W.WBW.W.W
W.W.WRW.W.W
W.........W
WWWWWOWWWWW
e.........e
WWWWW.WWWWW
r.........e
WWWWWZWWWWW
gR.......M.
WWWWWWWWWWB
WWWWWWWWWWg
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["up", "up", "up", "up", "up", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["up", "right", "right", "right", "right", "right", "left", "left", "left", "left", "up", "up", "left", "left", "left", "left", "left", "right", "right", "right", "right", "up", "up", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["up", "left", "left", "left", "left", "left", "right", "right", "right", "right", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["up", "left", "left", "left", "left", "left", "right", "right", "right", "right", "down", "down", "left", "left", "left", "left", "left"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    },
    2: {
      movements: {
        experienced1: { 
          path: ["up", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["up", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "up", "up", "up", "down", "down", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["down", "down", "down", "left", "left", "left", "left", "left", "right", "right", "right", "right", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["down", "down", "down", "left", "left", "left", "left", "left", "right", "right", "right", "right", "down", "down", "left", "left", "left", "left", "left"],
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