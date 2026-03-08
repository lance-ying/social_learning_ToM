import { LevelConfig } from '../types';

export const s212: LevelConfig = {
  id: 's212',
  name: 's212',
  asciiMap: `
WWeWeWWWWWg
WW.W.WWWWWB
b..Z..O....
WWW.WWW.WWW
e...WWW.WWW
WWW.WWW.WWW
WWW.WWW.WWW
.......M..r
RWW.WWW.WWW
.WW.WWW.WWW
.WW.WWW..Bg
gWW.WWWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["left", "left", "left", "right", "right", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "down", "down", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["left", "up", "up", "down", "left", "left", "right", "right", "down", "down", "down", "down", "down", "right", "right", "right", "right", "down", "down", "down", "right", "right", "right"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    },
    2: {
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