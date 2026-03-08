import { LevelConfig } from '../types';

export const old_s532: LevelConfig = {
  id: 'old_s532',
  name: 'old_s532',
  asciiMap:`
eWWe.....bW
.WWWWW.WWWW
.WWWWW.WWWW
....Z....eW
WWWW.WWWWWW
gR...WWWWWW
WWWW.WWWWWW
WWWWMWWWWWW
........BGW
.WWWBWWWWWW
rWWWgWWWWWW
WWWWWWWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: {
          path: ["right", "right", "up", "up", "up", "right", "right", "right", "left", "left", "down", "down", "down", "left", "left", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right"],
          goal: 2,
          type: 'Expert'
        },
        experienced2: {
          path: ["down", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "up", "up", "up", "left", "left", "left", "left"],
          goal: 1,
          type: 'Novice'
        },
        experienced3: {
          path: ["right", "right", "up", "up", "up", "right", "right", "right", "left", "left", "down", "down", "down", "left", "left", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right"],
          goal: 2,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["down", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "up", "up", "up", "left", "left", "left", "left"],
          goal: 1,
          type: 'Novice_2'
        },
      }
    },
  },
  stepsRemaining: 115,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
