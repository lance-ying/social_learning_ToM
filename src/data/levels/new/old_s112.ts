import { LevelConfig } from '../types';

export const old_s112: LevelConfig = {
  id: 'old_s112',
  name: 'old_s112',
  asciiMap: `
WWWWWWWWWWW
WeWWWWWg..W
W.WWWWWWW.W
W.........W
WWWWWZWWWWW
We.......eW
WWWWW.WWWWW
WgBR.MWWWWW
WWWWW.WWWWW
W.......BgW
W.WWW.WWWWW
WrWWWbWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["down", "down", "down", "down", "down", "down", "down", "up", "left", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "up", "up", "left", "left", "left", "left"],
          goal: 2,
          type: 'Expert'
        },
        experienced2: { 
          path: ["down", "down", "down", "down", "down", "down", "down", "up", "left", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "up", "up", "left", "left", "left", "left"],
          goal: 2,
          type: 'Expert'
        },
        experienced3: { 
          path: ["down", "down", "down", "down", "down", "down", "down", "up", "left", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "up", "up", "left", "left", "left", "left"],
          goal: 2,
          type: 'Expert'
        },
        experienced4: { 
          path: ["down", "down", "down", "down", "down", "down", "down", "up", "left", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "up", "up", "left", "left", "left", "left"],
          goal: 2,
          type: 'Expert'
        }
      }
    }
  },
  stepsRemaining: 75,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};