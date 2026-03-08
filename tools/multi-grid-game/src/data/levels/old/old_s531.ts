import { LevelConfig } from '../types';

export const old_s531: LevelConfig = {
  id: 'old_s531',
  name: 'old_s531',
  asciiMap: `
eWWWe.....b
.WWWWWW.WWW
.WWWWWW.WWW
..........e
WWWWW.WWWWW
g....Z.....
WWWWW.WWWWW
WWWWWMWWWWW
WWWWW.WWWWW
.........Bg
.WWWWBWWWWW
rWWWWgWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["up", "up", "right", "right", "up", "up", "up", "right", "right", "left", "left", "down", "down", "down", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["left", "left", "left", "left", "left"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["up", "up", "right", "right", "up", "up", "up", "right", "right", "left", "left", "down", "down", "down", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "left", "down", "up", "right", "right", "right", "right", "right", "up", "up", "up", "up", "left", "left", "left", "left"],
          goal: 2,
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