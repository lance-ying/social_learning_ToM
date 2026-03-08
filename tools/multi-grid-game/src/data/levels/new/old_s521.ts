import { LevelConfig } from '../types';

export const old_s521: LevelConfig = {
  id: 'old_s521',
  name: 'old_s521',
  asciiMap:`
bWWWWWWWWWe
.WWWWWWWWW.
.WWWWWWWWW.
...........
WWWW.WWWWWW
WWWW......e
WWWW.WWWWWW
gWWW......e
BWWW.WWWWWW
M...Z.....e
BWWW.WWWWWW
GWWW......g
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: {
          path: ["up", "up", "up", "up", "up", "up", "left", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Expert'
        },
        experienced2: {
          path: ["down", "down", "right", "right", "right", "right", "right", "right"],
          goal: 3,
          type: 'Novice'
        },
        experienced3: {
          path: ["up", "up", "up", "up", "up", "up", "left", "left", "left", "left", "up", "up", "up", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["down", "down", "right", "right", "right", "right", "right", "right"],
          goal: 3,
          type: 'Novice_2'
        },
      }
    },
  },
  stepsRemaining: 130,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
