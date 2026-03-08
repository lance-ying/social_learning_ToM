import { LevelConfig } from '../types';

export const sm332: LevelConfig = {
  id: 'sm332',
  name: 'sm332',
  asciiMap: `
WWWWeWWWWWg
WWWW.WWWWWB
b......O...
WWW.WWWWWWW
WWW.WWWWWWW
WWWZWWWWWWW
WWW.WWWWWWW
..........r
RWW.WWW.WWW
.WW.WWW.WWW
.WW.WWW..BG
gWW..MWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: {
          path: ["up", "up", "up", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: {
          path: ["down", "down", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced3: {
          path: ["up", "up", "up", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["down", "down", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down"],
          goal: 3,
          type: 'Novice_2'
        }
      }
    },
    2: {
      movements: {
        experienced1: {
          path: ["left", "left", "left", "left", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: {
          path: ["left", "left", "left", "left", "left", "left", "left", "right", "right", "down", "down", "down", "down", "down", "right", "right", "right", "right", "down", "down", "down", "right", "right", "right"],
          goal: 2,
          type: 'Expert'
        },
        experienced3: {
          path: ["left", "left", "left", "left", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down"],
          goal: 3,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["left", "left", "left", "left", "left", "left", "left", "right", "right", "down", "down", "down", "down", "down", "right", "right", "right", "right", "down", "down", "down", "right", "right", "right"],
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
