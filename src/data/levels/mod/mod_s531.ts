import { LevelConfig } from '../types';

export const mod_s531: LevelConfig = {
  id: 'mod_s531',
  name: 'mod_s531',
  asciiMap: `
eWWWe.....b
.WWWWWW.WWW
.WWWWWW.WWW
..........e
WWWWW.WWWWW
WWWWWZ.....
WWWWW.WWWWW
WWWWWMWWWWW
WWWWW.WWWWW
.........Bg
.WWWWWWWWWW
.WWWWWWWWWW
`.trim(),
  agentPaths: {
    2: {
      movements: {
        experienced1: { 
          path: ["up", "up", "right", "right", "up", "up", "up", "right", "right", "right", "left", "left", "down", "down", "down", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["up", "up", "right", "right", "up", "up", "up", "left", "left", "left", "right", "right", "right", "right", "right", "left", "left", "down", "down", "down", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: [],
          goal: 0,
          type: 'Expert_2'
        },
        experienced4: { 
          path: [],
          goal: 0,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 120,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
};