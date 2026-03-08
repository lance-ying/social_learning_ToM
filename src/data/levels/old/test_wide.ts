import { LevelConfig } from '../types';

export const test_wide: LevelConfig = {
  id: 'test_wide',
  name: 'Test Wide Map',
  asciiMap: `
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WeWWWrWWWeWWWWWWWWWWWWWWWWWWWW
W.WWW.WWW.WWWWWWWWWWWWWWWWWWWW
W.........................WWWW
WWWWWZWWWWWWWWWWWWWWWWWWWWWWWW
We.......bWWWWWWWWWWWWWWWWWWWW
WWWWW.WWWWWWWWWWWWWWWWWWWWWWWW
WgBR.M....................WWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["down", "down", "down", "down", "down", "down", "down"],
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
  stepsRemaining: 50,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B in this wide map'
  }
};