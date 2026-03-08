import { LevelConfig } from '../types';

export const sm531: LevelConfig = {
  id: 'sm531',
  name: 'sm531',
  asciiMap: `
eWWWWWbWWWW
.WWWWW.WWWW
.WWWWW.WWWW
..........e
WWWW.WWW.WW
gR..YWWW.WW
WWWW.WWW.WW
WWWW.WWW.WW
X........BG
.WWW.WWWMWW
.WWWBWWW.WW
rWWWgWWW.WW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["down", "down", "down", "up", "up", "right", "right", "right", "right", "up", "up", "up", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right", "right", "right", "up", "up", "up", "up", "up", "right", "right", "up", "up", "up", "down", "down", "right", "right", "down", "down", "down", "down", "down", "right", "right"],
          goal: 2,
          type: 'Expert'
        },
        experienced3: {
          path: ["down", "down", "down", "up", "up", "right", "right", "right", "right", "up", "up", "up", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["right", "right", "right", "right", "up", "up", "up", "up", "up", "right", "right", "up", "up", "up", "down", "down", "right", "right", "down", "down", "down", "down", "down", "right", "right"],
          goal: 2,
          type: 'Expert_2'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: ["up", "up", "right", "right", "up", "up", "up", "down", "down", "left", "left", "down", "down", "down", "down", "down", "down", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["down", "down", "down", "left", "left", "left", "left", "down", "down", "down", "up", "up", "right", "right", "right", "right", "up", "up", "up", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced3: {
          path: ["up", "up", "right", "right", "up", "up", "up", "down", "down", "left", "left", "down", "down", "down", "down", "down", "down", "down", "down"],
          goal: 3,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["down", "down", "down", "left", "left", "left", "left", "down", "down", "down", "up", "up", "right", "right", "right", "right", "up", "up", "up", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert_2'
        }
      }
    }
  },
  stepsRemaining: 100,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
