import { LevelConfig } from '../types';

export const sm531: LevelConfig = {
  id: 'sm531',
  name: 'sm531',
  asciiMap: `
eWWWbWWWWWW
.WWW.WWWWWW
O........WW
WWWW.WWWWWW
r...ZWWWWWW
WWWW.WWWWWW
WWWW.WWWWWW
.........BG
.WWW.WWW.WW
.WWW.WWW.WW
.WWWBWWWRWW
MWWWgWWWgWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["up", "up", "up", "up", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["left", "left", "left", "left", "right", "right", "right", "down", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced3: {
          path: [],
          goal: 3,
          type: 'Expert'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: ["up", "up", "down", "right", "right", "right", "right", "up", "up", "down", "down", "down", "down", "down", "down", "down", "down", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced2: {
          path: ["up", "up", "down", "right", "right", "right", "right", "up", "up", "down", "down", "down", "down", "down", "down", "down", "down", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: [],
          goal: 2,
          type: 'Novice'
        }
      }
    }
  },
  stepsRemaining: 105,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
