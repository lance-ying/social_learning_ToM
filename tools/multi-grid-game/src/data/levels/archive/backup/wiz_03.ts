import { LevelConfig } from '../types';

export const wiz_03: LevelConfig = {
  id: 'wiz_03',
  name: 'Double wizard sequence',
  asciiMap:`
rWWWWWWWWbW
.WW.....W.W
.WW.WWW.W.W
.WW.WWWeW.W
....WWW...W
WWW.WWW.WWW
WWM.WWW.WWW
WW..WWW..ZW
WWRBWWWWWWW
WW.BWWWWWWW
WW.......gW
WWWWWWWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["left", "left", "up", "up", "up", "up", "up", "up", "left", "left", "left", "left", "down", "down", "down", "left", "left", "left", "up", "up", "up", "up", "down", "down", "down", "down", "right", "right", "right", "down", "down", "down", "left", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["left", "left", "up", "up", "up", "up", "up", "up", "left", "left", "left", "left", "down", "down", "down", "left", "left", "left", "up", "down", "right", "left", "up", "down", "right", "left", "right", "right", "right", "left", "left", "left", "right", "left", "right", "right", "left", "right", "left", "left", "up", "down", "up", "down", "up", "up", "down", "up", "down", "down", "right", "left", "right", "right", "right", "up", "up", "up", "right", "left", "right", "left", "down", "down", "down", "up", "down", "up", "down", "up", "up", "down", "up", "up", "right", "left", "right", "left", "down", "up", "down", "up", "down", "down", "up", "down", "down", "left", "left", "right", "left", "left", "up", "up", "up", "up", "down", "down", "up", "down", "down", "up", "down", "down", "right", "right", "left", "left", "right", "left", "up", "up", "up", "down", "up", "up", "down", "up", "down", "up", "down", "up", "down", "down", "up", "down", "down", "down", "right", "left", "up", "up", "down", "up", "up", "up", "down", "up", "down", "down", "down", "up", "down", "down", "right", "left", "up", "up", "up", "up"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["left", "left", "up", "up", "up", "up", "up", "up", "left", "left", "left", "left", "down", "down", "down", "left", "left", "left", "up", "up", "up", "up", "down", "down", "down", "down", "right", "right", "right", "down", "down", "down", "left", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["left", "left", "right", "right", "left", "left", "up", "up", "up", "up", "down", "right", "right", "up", "up", "up", "up", "down", "down", "down", "down", "up", "down", "up", "down", "left", "right", "left", "left", "up", "down", "right", "right", "up", "down", "left", "left", "right", "right", "up", "up", "up", "down", "up", "up", "down", "down", "down", "up", "up", "up", "down", "down", "down", "up", "down", "up", "up", "up", "down", "down", "up", "down", "down", "up", "down", "up", "up", "down", "up", "down", "up", "up", "down", "down", "up", "down", "down", "down", "up", "up", "down", "up", "up", "down", "down", "down", "up", "up", "down", "up", "down", "down", "up", "down", "up", "down", "left", "left", "up", "up", "up", "down", "up", "left", "left", "left", "left", "right", "right", "left", "right", "right", "right", "down", "up", "down", "down", "up", "up", "left", "right", "left", "left", "right", "left", "left", "left", "right", "left", "right", "right", "right", "right", "left", "right", "left", "right", "left", "right", "left", "right", "left", "left", "left", "right", "right", "right", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        },
      }
    },
  },
  stepsRemaining: 50,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};
