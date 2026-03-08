import { LevelConfig } from "../types";

export const sm_demo2: LevelConfig = {
  id: "sm_demo2",
  name: "sm_demo2",
  asciiMap: `
WWWWWWWWWWWW
WWWWWWWWWWWW
WWWWWWWWWWWW
WWWgWrWeWbWW
WWWRW.W.W.WW
WWW.W.WO..WW
WgBZ....W.WW
WWWWBWWWW.WW
WWWWgWWWWMWW
WWWWWWWWWWWW
WWWWWWWWWWWW
`.trim(),
  agentPaths: {
    3: {
      movements: {
        experienced1: {
          path: ["up", "up", "down", "right", "right", "up", "up", "down", "left", "left", "down","left", "left", "left", "down", "down"],
          goal: 1,
          type: "Novice",
        },
        experienced2: {
          path: ["up", "up", "down", "right", "right", "up", "up", "down", "left", "left", "down","left", "left", "left", "down", "down"],
          goal: 3,
          type: "Novice",
        },
        experienced3: {
          path: ["up", "up", "down", "right", "right", "up", "up", "down", "left", "left", "down","left", "left", "left", "down", "down"],
          goal: 3,
          type: "Novice",
        },
      },
    },
    2: {
      movements: {
        experienced1: {
          path: ["right", "right", "right", "right", "up", "right", "right", "up", "up", "down", "left", "left", "down", "left", "left", "left", "left", "left", "left"],
          goal: 2,
          type: "Expert",
        },
        experienced2: {
          path: ["right", "right", "right", "right", "up", "right", "right", "up", "up", "down", "left", "left", "down", "left", "left", "left", "left", "left", "left"],
          goal: 1,
          type: "Expert",
        },
        experienced3: {
          path: ["right", "right", "right", "right", "up", "right", "right", "up", "up", "down", "left", "left", "down", "left", "left", "left", "left", "left", "left"],
          goal: 2,
          type: "Expert",
        },
      },
    },
  },
  stepsRemaining: 65,
  goal: {
    type: "C",
    description: "Find and obtain Treasure C",
  },
};
