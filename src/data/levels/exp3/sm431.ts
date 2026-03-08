import { LevelConfig } from '../types';

export const sm431: LevelConfig = {
  id: 'sm431',
  name: 'sm431',
  asciiMap: `
eWeWbWWWWgW
.W.W.WWWWBW
.W.W.WWWW.W
.....WWWW.W
.WWWWWWWW.W
..Z..O.....
WWW.WWWWWW.
WWW.WWWWWWe
WWW.WWWWWWW
.....M...rW
BWWWRWWWWWW
GWWWgWWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: {
          path: ["left", "left", "up", "up", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced3: {
          path: ["left", "left", "up", "up", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "up", "up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down"],
          goal: 3,
          type: 'Novice_2'
        }
      }
    },
    2: {
      movements: {
        experienced1: {
          path: ["left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["left", "left", "left", "left", "left", "up", "up", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "down", "down", "down", "down", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Expert'
        },
        experienced3: {
          path: ["left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down"],
          goal: 3,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["left", "left", "left", "left", "left", "up", "up", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "down", "down", "right", "right", "right", "down", "down", "down", "down", "left", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 50,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
