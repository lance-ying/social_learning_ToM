import { LevelConfig } from '../types';

export const s0001: LevelConfig = {
  id: 's0001',
  name: 's0001',
  asciiMap:`
WWrWeWbW
WW.W.W.W
gR.W.W.W
WW....M.
gB.WZWWW
`.trim(),
  agentPaths: {
    1: {
      movements: {
        experienced1: { 
          path: ["up", "up", "up", "up", "up", "up", "up", "right", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["up", "up", "up", "up", "up", "up", "up", "right", "right", "up", "up", "up", "down", "down", "right", "right", "up", "up", "up", "down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["up", "up", "up", "up", "up", "up", "up", "up", "up", "up", "down", "down", "right", "right", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["up", "up", "up", "up", "up", "up", "up", "up", "up", "up", "down", "down", "right", "right", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    },
    2: {
      movements: {
        experienced1: { 
          path: ["up", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert'
        },
        experienced2: { 
          path: ["up", "right", "right", "up", "up", "up", "down", "down", "right", "right", "up", "up", "up", "down", "down", "right", "right", "right", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right"],
          goal: 2,
          type: 'Novice'
        },
        experienced3: {
          path: ["up", "left", "left", "up", "up", "up", "down", "down", "right", "right", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right"],
          goal: 1,
          type: 'Expert_2'
        },
        experienced4: {
          path: ["up", "left", "left", "up", "up", "up", "down", "down", "right", "right", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right"],
          goal: 2,
          type: 'Novice_2'
        }
      }
    }
  },
  stepsRemaining: 50,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
}; 