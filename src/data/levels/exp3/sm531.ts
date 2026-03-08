import { LevelConfig } from '../types';

export const sm531: LevelConfig = {
  id: 'sm531',
  name: 'sm531',
  asciiMap: `
eWWWWWbWWWW
.WWWWW.WWWW
.WWWWW.WWWW
..........e
WWWW.WWWWWW
gR..ZWWWWWW
WWWW.WWWWWW
WWWW.WWWWWW
O....M..BGW
.WWWBWWWWWW
rWWWgWWWWWW
WWWWWWWWWWW
`.trim(),
  agentPaths: {
    1: {
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
        }
      }
    },
    2: {
      movements: {
        experienced1: {
          path: ["down", "down", "up", "right", "right", "right", "right", "up", "up", "up", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right", "right", "right", "up", "up", "up", "up", "up", "right", "right", "up", "up", "up", "down", "down", "left", "left", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right"],
          goal: 2,
          type: 'Expert'
        },
        experienced3: {
          path: ["down", "down", "up", "right", "right", "right", "right", "up", "up", "up", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["right", "right", "right", "right", "up", "up", "up", "up", "up", "right", "right", "up", "up", "up", "down", "down", "left", "left", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 50,
  goal: {
    type: 'C',
    description: 'Find and obtain Treasure C'
  }
};
