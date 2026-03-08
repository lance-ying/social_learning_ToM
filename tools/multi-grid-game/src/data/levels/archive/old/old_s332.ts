import { LevelConfig } from '../types';

export const old_s332: LevelConfig = {
  id: 'old_s332',
  name: 'old_s332',
  asciiMap: `
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
.WW.WWW..Bg
gWWMWWWWWWW
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
          path: ["down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["up", "up", "up", "right", "up", "down", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 40,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};