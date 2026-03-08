import { LevelConfig } from '../types';

export const sm622: LevelConfig = {
  id: 'sm622',
  name: 'sm622',
  asciiMap: `
bWWWWWWWWW.
.WWWWWWWWW.
...........
WW.WWWWWWW.
WW.WWWWWWW.
WW........O
WW.WWWWWWW.
gW.WWWWWWWe
BW.WWWWWWWW
M.........Z
BWWWWWWWWW.
GWWWWWWWWWg
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["left", "left", "left", "left", "left", "left", "left", "left", "up", "up", "up", "up", "up", "up", "up", "left", "left", "up", "up", "down", "right", "right", "down", "down", "down", "down", "down", "down", "down", "left", "left", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["left", "left", "left", "left", "left", "left", "left", "left", "up", "up", "up", "up", "up", "up", "up", "left", "left", "up", "up", "down", "right", "right", "down", "down", "down", "down", "down", "down", "down", "left", "left", "down", "down"],
          goal: 2,
          type: 'Expert'
        },
        experienced3: {
          path: [],
          goal: 2,
          type: 'Expert'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: ["left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Novice'
        },
        experienced2: {
          path: ["left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right", "down", "down"],
          goal: 3,
          type: 'Novice'
        },
        experienced3: {
          path: [],
          goal: 3,
          type: 'Novice'
        }
      }
    }
  },
  stepsRemaining: 105,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
