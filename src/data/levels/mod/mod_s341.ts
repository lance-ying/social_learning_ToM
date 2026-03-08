import { LevelConfig } from '../types';

export const mod_s341: LevelConfig = {
  id: 'mod_s341',
  name: 'mod_s341',
  asciiMap: `
WWWWWWWWW
WWWWWWWWW
WWWrWbWeW
WWW.W.W.W
WWW.W.W.W
WWW....MW
WgB.WZWWW
WWWWWWWWW
WWWWWWWWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: { 
          path: ["up", "up", "up", "up", "down", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["left", "left", "left", "left", "left", "left", "up", "up", "down", "down", "down", "up", "right", "right", "right", "down", "down", "down", "down", "down", "right", "right", "down", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["left", "left", "left", "left", "left", "left", "down", "down", "up", "right", "right", "right", "down", "down", "down", "down", "down", "right", "right", "down", "down", "down"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["left", "left", "left", "left", "left", "left", "up", "up", "down", "down", "down", "up", "right", "right", "right", "down", "down", "down", "down", "down", "right", "right", "down", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },

  stepsRemaining: 125,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};