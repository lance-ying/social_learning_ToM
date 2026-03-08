import { LevelConfig } from '../types';

export const sm111: LevelConfig = {
  id: 'sm111',
  name: 'sm111',
  asciiMap: `
WWWWWWWWWWW
WeWWWeWWWbW
W.X.......W
WWWWW.WWWWW
W........rW
WWWWW.WWWWW
WgBR.MWWWWW
WWWWW.WWWWW
WY......BGW
WWWWW.WWWWW
WWWWWgWWWWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ['left', 'up', 'right', 'right', 'right', 'right', 'up', 'right', 'right', 'right', 'right', 'up', 'left', 'left', 'left', 'left', 'down', 'down', 'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right'],
          goal: 3,
          type: 'Novice'
        },
        experienced2: {
          path: ['left', 'up', 'right', 'right', 'right', 'right', 'up', 'right', 'right', 'right', 'right', 'up', 'left', 'left', 'left', 'left', 'down', 'down', 'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right'],
          goal: 3,
          type: 'Novice'
        },
        experienced3: {
          path: ['left', 'up', 'right', 'right', 'right', 'right', 'up', 'right', 'right', 'right', 'right', 'up', 'left', 'left', 'left', 'left', 'down', 'down', 'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right'],
          goal: 3,
          type: 'Novice'
        },
        experienced4: {
          path: ['left', 'up', 'right', 'right', 'right', 'right', 'up', 'right', 'right', 'right', 'right', 'up', 'left', 'left', 'left', 'left', 'down', 'down', 'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right'],
          goal: 3,
          type: 'Novice'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: ['right', 'right', 'right', 'right', 'up', 'up', 'up', 'up', 'up', 'up', 'right', 'right', 'right', 'right', 'up', 'left', 'left', 'left', 'left', 'down', 'down', 'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right'],
          goal: 2,
          type: 'Expert'
        },
        experienced2: {
          path: ['right', 'right', 'right', 'right', 'up', 'up', 'up', 'up', 'up', 'up', 'right', 'right', 'right', 'right', 'up', 'left', 'left', 'left', 'left', 'down', 'down', 'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right'],
          goal: 2,
          type: 'Expert'
        },
        experienced3: {
          path: ['right', 'right', 'right', 'right', 'up', 'up', 'up', 'up', 'up', 'up', 'right', 'right', 'right', 'right', 'up', 'left', 'left', 'left', 'left', 'down', 'down', 'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right'],
          goal: 2,
          type: 'Expert'
        },
        experienced4: {
          path: ['right', 'right', 'right', 'right', 'up', 'up', 'up', 'up', 'up', 'up', 'right', 'right', 'right', 'right', 'up', 'left', 'left', 'left', 'left', 'down', 'down', 'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right'],
          goal: 2,
          type: 'Expert'
        }
      }
    }
  },
  stepsRemaining: 90,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
