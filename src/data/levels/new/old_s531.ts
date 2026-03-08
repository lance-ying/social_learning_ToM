import { LevelConfig } from '../types';

export const old_s531: LevelConfig = {
  id: 'old_s531',
  name: 'old_s531',
  asciiMap:`
eWWe.....bW
.WWWWW.WWWW
.WWWWW.WWWW
.........eW
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
    1: {
      movements: {
        experienced1: {
          path: ["up", "up", "right", "right", "up", "up", "up", "right", "right", "right", "left", "left", "down", "down", "down", "left", "left", "down", "down", "down", "down", "down", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["down", "down", "down", "left", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "up", "up", "up", "left", "left", "left", "left"],
          goal: 1,
          type: 'Novice'
        },
        experienced3: {
          path: ["up", "up", "right", "right", "up", "up", "up", "right", "right", "right", "left", "left", "down", "down", "down", "left", "left", "down", "down", "down", "down", "down", "down", "down"],
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
  stepsRemaining: 110,
  goal: {
    type: 'C',
    description: 'Find and obtain Treasure C'
  }
};
