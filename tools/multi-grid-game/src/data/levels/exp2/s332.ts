import { LevelConfig } from '../types';

export const s332: LevelConfig = {
  id: 's332',
  name: 's332',
  asciiMap:`
WWWWeWWWWWg
WWWW.WWWWWB
b..........
WWW.WWWWWWW
WWW.WWWWWWW
WWWZWWWWWWW
WWW.WWWWWWW
..........r
RWW.WWW.WWW
.WW.WWW.WWW
.WW.WWW..BG
gWWMWWWWWWW

`.trim(),
  agentPaths: {
    2: {
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
        },
      }
    },
  },
  stepsRemaining: 115,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};
