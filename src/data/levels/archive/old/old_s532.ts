import { LevelConfig } from '../types';

export const old_s532: LevelConfig = {
  id: 'old_s532',
  name: 'old_s532',
  asciiMap: `
eWWWe.....b
.WWWWWW.WWW
.WWWWWW.WWW
.......Z..e
WWWWW.WWWWW
g..........
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
          path: ["up", "up", "up", "right", "right", "right", "left", "left", "down", "down", "down", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["left", "left", "down", "down", "left", "left", "left", "left", "left"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: { 
          path: ["up"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: { 
          path: ["up", "up", "up", "right", "right", "left", "left", "down", "down", "down", "left", "left", "down", "down", "down", "down", "down", "down", "down", "down"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 65,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
};