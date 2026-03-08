import { LevelConfig } from '../types';

export const old_s111: LevelConfig = {
  id: 'old_s111',
  name: 'old_s111',
  asciiMap: `
WWWWWWWWWWW
WeWWWrWWWeW
W.WWW.WWW.W
W.........W
WWWWWZWWWWW
We.......bW
WWWWW.WWWWW
WgBR.MWWWWW
WWWWW.WWWWW
W.......BgW
WWWWW.WWWWW
WWWWWgWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["down", "right", "right", "right", "right", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right"],
          goal: 3,
          type: 'Expert'
        },
        experienced2: { 
          path: ["down", "down", "down", "down", "down", "down", "down"],
          goal: 3,
          type: 'Novice'
        },
        experienced3: { 
          path: ["down", "down", "down", "down", "down", "down", "down"],
          goal: 3,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["down", "down", "down", "down", "down", "down", "down"],
          goal: 3,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 70,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};