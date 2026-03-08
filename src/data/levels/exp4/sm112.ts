import { LevelConfig } from '../types';

export const sm112: LevelConfig = {
  id: 'sm112',
  name: 'sm112',
  asciiMap: `
WWWWWWWWWWW
WeWeWWWWWWW
W....Y....W
WWWWW.WWWWW
WWWWWZ...bW
WWWWW.WWWWW
WGBR.MWWWWW
WWWWW.WWWWW
W.......BgW
W.WWWRWWWWW
WrWWWgWWWWW
`.trim(),
  agentPaths: {
    3: {
      movements: {
        experienced1: {
          path: ['left', 'left', 'up', 'left', 'left', 'up', 'right', 'right', 'right', 'right', 'down', 'down', 'right', 'right', 'right', 'right', 'left', 'left', 'left', 'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right'],
          goal: 1,
          type: 'Novice'
        },
        experienced2: {
          path: ['left', 'left', 'up', 'left', 'left', 'up', 'right', 'right', 'right', 'right', 'down', 'down', 'right', 'right', 'right', 'right', 'left', 'left', 'left', 'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right'],
          goal: 1,
          type: 'Novice'
        },
        experienced3: {
          path: ['left', 'left', 'up', 'left', 'left', 'up', 'right', 'right', 'right', 'right', 'down', 'down', 'right', 'right', 'right', 'right', 'left', 'left', 'left', 'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right'],
          goal: 1,
          type: 'Novice'
        },
        experienced4: {
          path: ['left', 'left', 'up', 'left', 'left', 'up', 'right', 'right', 'right', 'right', 'down', 'down', 'right', 'right', 'right', 'right', 'left', 'left', 'left', 'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right'],
          goal: 1,
          type: 'Novice'
        }
      }
    },
    2: {
      movements: {
        experienced1: {
          path: ['down', 'down', 'down', 'down', 'left', 'left', 'left', 'left', 'left', 'down', 'down', 'up', 'right', 'right', 'right', 'right', 'down', 'down'],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ['down', 'down', 'down', 'down', 'left', 'left', 'left', 'left', 'left', 'down', 'down', 'up', 'right', 'right', 'right', 'right', 'down', 'down'],
          goal: 3,
          type: 'Expert'
        },
        experienced3: {
          path: ['down', 'down', 'down', 'down', 'left', 'left', 'left', 'left', 'left', 'down', 'down', 'up', 'right', 'right', 'right', 'right', 'down', 'down'],
          goal: 3,
          type: 'Expert'
        },
        experienced4: {
          path: ['down', 'down', 'down', 'down', 'left', 'left', 'left', 'left', 'left', 'down', 'down', 'up', 'right', 'right', 'right', 'right', 'down', 'down'],
          goal: 3,
          type: 'Expert'
        }
      }
    }
  },
  stepsRemaining: 115,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
