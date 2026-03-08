import { LevelConfig } from '../types';

export const s211: LevelConfig = {
  id: 's211',
  name: 's211',
  asciiMap: `
WWeWeWWWWWg
WW.W.WWWWWB
b..M..Z....
WWW.WWW.WWW
e...WWW.WWW
WWW.WWW.WWW
WWW.WWW.WWW
...O......r
RWW.WWW.WWW
.WW.WWW.WWW
.WW.WWW..Bg
gWW.WWWWWWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: { 
          path: ["up", "up", "up", "up", "up", "left", "left", "left", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "down", "down", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["up", "up", "up", "left", "left", "left", "right", "right", "up", "up", "left", "left", "left", "right", "right", "down", "down", "down", "down", "down", "right", "right", "right", "right", "down", "down", "down", "right", "right", "right"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    },
    1: {
      movements: {
        experienced1: { 
          path: ["left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "down", "down", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["left", "left", "up", "up", "down", "left", "left", "up", "up", "down", "left", "left", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "down", "down", "right", "right", "right"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["right", "down", "down", "down", "down", "down", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["right", "down", "down", "down", "down", "down", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down"],
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