import { LevelConfig } from '../types';

export const s311: LevelConfig = {
  id: 's311',
  name: 's311',
  asciiMap: `
eWWeWWeWWWb
.WW.WW.WWW.
.WW.WW.WWW.
M....Z.....
WW.WWW.WWW.
r.OWWW.WWW.
WW.WWWeWWWe
WW.WWWWWWWW
WW.WWWWWWWW
.........Rg
BWWWBWWWWWW
gWWWgWWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["right", "up", "up", "up", "down", "down", "down", "down", "down", "up", "up", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["left", "left", "left", "down", "down", "left", "left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["left", "left", "left", "down", "down", "left", "left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    },
    2: {
      movements: {
        experienced1: { 
          path: ["up", "up", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "down", "down"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["up", "up", "right", "up", "up", "up", "down", "down", "right", "right", "right", "up", "up", "up", "down", "down", "down", "down", "down", "up", "up", "right", "right", "right", "right", "up", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "down", "down"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["left", "left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["left", "left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"],
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