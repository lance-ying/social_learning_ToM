import { LevelConfig } from '../types';

export const sm361: LevelConfig = {
  id: 'sm361',
  name: 'sm361',
  asciiMap: `
eWWWWWWeWWb
.WWWWWW.WW.
.......Z...
WWWW.WWW.WW
WWWW.WWW.WW
WWWW.WWW.WW
WWWW.WWW.WW
r...OWWW.WW
WWWW.WWW.WW
..........M
RWWW.WW.WWW
gWWWgWW.B.G
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: {
          path: ["up", "up", "down", "right", "right", "right", "up", "up", "down", "left", "left", "down", "down", "down", "down", "down", "down", "down", "left", "down", "down", "right", "right", "right"],
          goal: 3,
          type: 'Novice'
        },
        experienced2: {
          path: ["left", "left", "left", "down", "down", "down", "down", "down", "left", "left", "left", "left", "right", "right", "right", "down", "down", "left", "left", "left", "left", "down", "down"],
          goal: 1,
          type: 'Novice'
        },
        experienced3: {
          path: ["up", "up", "down", "right", "right", "right", "up", "up", "down", "left", "left", "down", "down", "down", "down", "down", "down", "down", "left", "down", "down", "right", "right", "right"],
          goal: 3,
          type: 'Novice'
        }
      }
    },
    3: {
      movements: {
        experienced1: {
          path: ["up", "up", "up", "up", "up", "right", "right", "right", "right", "right", "right", "up", "up", "down", "left", "left", "down", "down", "down", "down", "down", "down", "down", "left", "down", "down", "right", "right", "right"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["down", "down", "down", "down"],
          goal: 2,
          type: 'Expert'
        },
        experienced3: {
          path: ["up", "up", "up", "up", "up", "right", "right", "right", "right", "right", "right", "up", "up", "down", "left", "left", "down", "down", "down", "down", "down", "down", "down", "left", "down", "down", "right", "right", "right"],
          goal: 3,
          type: 'Expert'
        }
      }
    }
  },
  stepsRemaining: 110,
  goal: {
    type: 'C',
    description: 'Find and obtain Treasure C'
  }
};
