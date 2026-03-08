import { LevelConfig } from '../types';

export const sm361: LevelConfig = {
  id: 'sm361',
  name: 'sm361',
  asciiMap: `
eWWWWWWeWbW
.WWWWWW.W.W
....Z..O...
WWWW.WWW.WW
WWWW.WWW.WW
WWWW.WWW.WW
WWWW.WWW.WW
r....WWW.WW
WWWW.WWW.WW
........M..
RWWW.WW.WWW
gWWWgWW.B.G
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["down", "down", "down", "down", "down", "down", "down", "down", "down"],
          goal: 2,
          type: 'Expert'
        },
        experienced2: {
          path: ["right", "right", "right", "right", "right", "up", "up", "down", "left", "down", "down", "down", "down", "down", "down", "down", "left", "down", "down", "right", "right", "right"],
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
          path: ["up", "up", "down", "right", "right", "up", "up", "down", "left", "down", "down", "down", "down", "down", "down", "down", "left", "down", "down", "right", "right", "right"],
          goal: 3,
          type: 'Novice'
        },
        experienced2: {
          path: ["up", "up", "down", "right", "right", "up", "up", "down", "left", "down", "down", "down", "down", "down", "down", "down", "left", "down", "down", "right", "right", "right"],
          goal: 3,
          type: 'Novice'
        },
        experienced3: {
          path: [],
          goal: 3,
          type: 'Novice'
        }
      }
    }
  },
  stepsRemaining: 95,
  goal: {
    type: 'C',
    description: 'Find and obtain Treasure C'
  }
};
