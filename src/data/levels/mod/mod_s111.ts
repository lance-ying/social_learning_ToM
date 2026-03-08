import { LevelConfig } from '../types';

export const mod_s111: LevelConfig = {
  id: 'mod_s111',
  name: 'mod_s111',
  asciiMap: `
WWWWWWWWWWW
WeWWW.WWWeW
W.WWW.WWW.W
W.........W
WWWWWZWWWWW
We.......bW
WWWWW.WWWWW
WWWWWMWWWWW
WWWWW.WWWWW
W.......BgW
WWWWWWWWWWW
WWWWWWWWWWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: { 
          path: ["down", "right", "right", "right", "right", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["down", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["down", "right", "right", "right", "right", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["down", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 85,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
