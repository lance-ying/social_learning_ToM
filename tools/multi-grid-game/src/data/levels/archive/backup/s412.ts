import { LevelConfig } from '../types';

export const s412: LevelConfig = {
  id: 's412',
  name: 's412',
  asciiMap: `
eWeWWrWWWgW
.W.WW.WWWRW
.W.WW.WWW.W
....O.....W
.WW.WWW.WWW
.WW.WWW.WWW
bWW.WWW.WWW
WWW.WWW.WWW
Z.....M....
WWWWWRWWW.W
WWWWWBWWW.W
WWWWWgWWWgW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["right", "right", "right", "up", "up", "up", "up", "up", "left", "left", "left", "down", "down", "down", "up", "up", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "down", "down", "down", "down", "down", "right", "right", "down", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["right", "right", "right", "up", "up", "up", "up", "up", "left", "up", "up", "up", "down", "down", "left", "left", "up", "up", "up", "down", "down", "down", "down", "down", "up", "up", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "down", "down", "down", "down", "down", "right", "right", "down", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["right", "right", "right", "right", "right", "right", "right", "right", "right", "down", "down", "down"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["right", "right", "right", "right", "right", "right", "right", "right", "right", "down", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    },
    2: {
      movements: {
        experienced1: { 
          path: ["left", "left", "left", "left", "left", "left", "left", "up", "up", "up", "up", "up", "left", "left", "left", "down", "down", "down", "up", "up", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "right", "right", "down", "down", "down", "down", "down", "left", "left", "down", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["left", "left", "left", "left", "left", "left", "left", "up", "up", "up", "up", "up", "left", "up", "up", "up", "down", "down", "left", "left", "up", "up", "up", "down", "down", "down", "down", "down", "up", "up", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "right", "right", "down", "down", "down", "down", "down", "left", "left", "down", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["left", "down", "down", "down"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["left", "down", "down", "down"],
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