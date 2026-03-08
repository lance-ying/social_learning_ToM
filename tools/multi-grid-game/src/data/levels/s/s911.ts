import { LevelConfig } from '../types';

export const s911: LevelConfig = {
  id: 's911',
  name: 's911',
  asciiMap:`
WWWWWWWWWWW
gB...O...Bg
WWWWW.WWWWW
WWWWW.....r
b.....WWWWW
WWWWW.....e
e....ZWWWWW
WWWWW.....e
e.....WWWWW
WWWWW.WWWWW
gR.......MW
WWWWWWWWWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["up", "up", "left", "left", "left", "left", "left", "right", "right", "right", "right", "up", "up", "up", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["left", "left", "left", "left", "left", "right", "right", "right", "right", "up", "right", "right", "right", "right", "right", "left", "left", "left", "left", "up", "left", "left", "left", "left", "left", "right", "right", "right", "right", "up", "up", "up", "left", "left", "left", "left", "left"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["up", "up", "up", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["up", "up", "up", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    },
    2: {
      movements: {
        experienced1: { 
          path: ["down", "down", "down", "left", "left", "left", "left", "left", "right", "right", "right", "right", "up", "up", "up", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["down", "down", "down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "left", "left", "left", "left", "left", "right", "right", "right", "right", "up", "up", "left", "left", "left", "left", "left", "right", "right", "right", "right", "up", "up", "up", "right", "right", "right", "right", "right"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "left"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "left"],
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