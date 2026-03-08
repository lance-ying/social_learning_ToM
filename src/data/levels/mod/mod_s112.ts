import { LevelConfig } from '../types';

export const mod_s112: LevelConfig = {
  id: 'mod_s112',
  name: 'mod_s112',
  asciiMap: `
WWWWWWWWWWW
WeWWWWWWWWW
W.WWWWWWWWW
W.........W
WWWWWZWWWWW
We.......eW
WWWWW.WWWWW
WWWWWMWWWWW
WWWWW.WWWWW
W......RBgW
W.WWW.WWWWW
WrWWWbWWWWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: { 
          path: ["down", "down", "down", "down", "down", "down", "down", "up", "left", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["down", "left", "left", "left", "left", "right", "right", "right", "down", "down", "down", "down", "down", "down", "up", "left", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["down", "down", "down", "down", "down", "down", "down", "up", "left", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["down", "left", "left", "left", "left", "right", "right", "right", "down", "down", "down", "down", "down", "down", "up", "left", "left", "left", "left", "down", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 95,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
