import { LevelConfig } from '../types';

export const sm531: LevelConfig = {
  id: 'sm531',
  name: 'sm531',
  asciiMap: `
eWWWWWbWWWW
.WWWWW.WWWW
O.........e
WWWW.WWW.WW
gR...WWW.WW
WWWW.WWW.WW
WWWW.WWW.WW
Z........BG
.WWW.WWW.WW
.WWW.WWWMWW
.WWWBWWW.WW
rWWWgWWW.WW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["down", "down", "down", "down", "up", "up", "up", "right", "right", "right", "right", "up", "up", "up", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right", "right", "right", "up", "up", "up", "up", "up", "right", "right", "up", "up", "down", "right", "right", "down", "down", "down", "down", "down", "right", "right"],
          goal: 2,
          type: 'Expert'
        },
        experienced3: {
          path: ["right", "right", "right", "right", "up", "up", "up", "up", "up", "right", "right", "up", "up", "down", "right", "right", "down", "down", "down", "down", "down", "right", "right"],
          goal: 2,
          type: 'Expert'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: ["up", "up", "down", "right", "right", "right", "right", "right", "right", "up", "up", "down", "left", "left", "down", "down", "down", "down", "down", "down", "down", "down", "down"],
          goal: 3,
          type: 'Novice'
        },
        experienced2: {
          path: ["right", "right", "right", "right", "down", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down", "down", "down", "up", "up", "up", "right", "right", "right", "right", "up", "up", "up", "left", "left", "left", "left"],
          goal: 1,
          type: 'Novice'
        },
        experienced3: {
          path: ["up", "up", "down", "right", "right", "right", "right", "right", "right", "up", "up", "down", "left", "left", "down", "down", "down", "down", "down", "down", "down", "down", "down"],
          goal: 3,
          type: 'Novice'
        }
      }
    }
  },
  stepsRemaining: 85,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
