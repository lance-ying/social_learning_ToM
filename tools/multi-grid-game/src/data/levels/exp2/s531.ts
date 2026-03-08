import { LevelConfig } from '../types';

export const s531: LevelConfig = {
  id: 's531',
  name: 's531',
  asciiMap:`
eWWWWWbWWWW
.WWWWW.WWWW
.WWWWW.WWWW
..........e
WWWW.WWWWWW
gR..ZWWWWWW
WWWW.WWWWWW
WWWW.WWWWWW
.....M..BGW
.WWWBWWWWWW
rWWWgWWWWWW
WWWWWWWWWWW

`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["up", "up", "right", "right", "up", "up", "up", "down", "down", "left", "left", "down", "down", "down", "down", "down", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["down", "down", "down", "left", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "up", "up", "up", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced3: {
          path: ["up", "up", "right", "right", "up", "up", "up", "down", "down", "left", "left", "down", "down", "down", "down", "down", "down", "down"],
          goal: 3,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["down", "down", "down", "left", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "up", "up", "up", "left", "left", "left", "left"],
          goal: 1,
          type: 'Novice_2'
        },
      }
    },
  },
  stepsRemaining: 100,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
