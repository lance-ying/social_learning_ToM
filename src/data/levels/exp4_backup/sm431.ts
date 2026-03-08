import { LevelConfig } from '../types';

export const sm431: LevelConfig = {
  id: 'sm431',
  name: 'sm431',
  asciiMap: `
eWeWbWeWWgW
.W.W.W.WWBW
.W.W.W.WW.W
Z......WW.W
.WWWWWWWW.W
...O.......
WWW.WWWWWWW
WWW.WWWWWWW
WWW.WWWWWWW
.....M...rW
BWWWRWWWWWW
GWWWgWWWWWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["up", "up", "up", "down", "down", "right", "right", "up", "up", "up", "down", "down", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "down", "down", "down", "down", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced2: {
          path: ["down", "down", "right", "right", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down"],
          goal: 3,
          type: 'Novice'
        },
        experienced3: {
          path: ["up", "up", "up", "down", "down", "right", "right", "up", "up", "up", "down", "down", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "down", "down", "down", "down", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: ["left", "left", "left", "up", "up", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced3: {
          path: ["left", "left", "left", "up", "up", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        }
      }
    }
  },
  stepsRemaining: 135,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
