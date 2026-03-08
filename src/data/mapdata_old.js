const mapData = {
level111: {
  player: { x: 5, y: 7 },
  npc: { 
    x: 5, 
    y: 4, 
    movements: {
      experienced1: { path: ["down", "down", "down", "down", "down", "down", "down"], goal: 3 },
      experienced2: { path: ["down", "down", "down", "down", "down", "down", "down"], goal: 3 },
      novice1: { path: ["down", "down", "down", "down", "down", "down", "down"], goal: 3 },
      novice2: { path: ["down", "down", "down", "down", "down", "down", "down"], goal: 3 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 2, y: 7, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 3, y: 7, requiredItems: ["redAmulet"], color: 'text-red-500' },
    { x: 8, y: 9, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 1, y: 7, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 9, y: 9, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 5, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 1, y: 1, content: 'nothing', color: 'text-blue-500' },
    { x: 5, y: 1, content: 'redAmulet', color: 'text-red-500' },
    { x: 9, y: 1, content: 'nothing', color: 'text-blue-500' },
    { x: 1, y: 5, content: 'nothing', color: 'text-blue-500' },
    { x: 9, y: 5, content: 'blueAmulet', color: 'text-blue-500' },
  ],
  blocks: [{x:0,y:0}, {x:1,y:0}, {x:2,y:0}, {x:3,y:0}, {x:4,y:0}, {x:5,y:0}, {x:6,y:0}, {x:7,y:0}, {x:8,y:0}, {x:9,y:0}, {x:10,y:0}, {x:0,y:1}, {x:2,y:1}, {x:3,y:1}, {x:4,y:1}, {x:6,y:1}, {x:7,y:1}, {x:8,y:1}, {x:10,y:1}, {x:0,y:2}, {x:2,y:2}, {x:3,y:2}, {x:4,y:2}, {x:6,y:2}, {x:7,y:2}, {x:8,y:2}, {x:10,y:2}, {x:0,y:3}, {x:10,y:3}, {x:0,y:4}, {x:1,y:4}, {x:2,y:4}, {x:3,y:4}, {x:4,y:4}, {x:6,y:4}, {x:7,y:4}, {x:8,y:4}, {x:9,y:4}, {x:10,y:4}, {x:0,y:5}, {x:10,y:5}, {x:0,y:6}, {x:1,y:6}, {x:2,y:6}, {x:3,y:6}, {x:4,y:6}, {x:6,y:6}, {x:7,y:6}, {x:8,y:6}, {x:9,y:6}, {x:10,y:6}, {x:0,y:7}, {x:6,y:7}, {x:7,y:7}, {x:8,y:7}, {x:9,y:7}, {x:10,y:7}, {x:0,y:8}, {x:1,y:8}, {x:2,y:8}, {x:3,y:8}, {x:4,y:8}, {x:6,y:8}, {x:7,y:8}, {x:8,y:8}, {x:9,y:8}, {x:10,y:8}, {x:0,y:9}, {x:10,y:9}, {x:0,y:10}, {x:1,y:10}, {x:2,y:10}, {x:3,y:10}, {x:4,y:10}, {x:6,y:10}, {x:7,y:10}, {x:8,y:10}, {x:9,y:10}, {x:10,y:10}, {x:0,y:11}, {x:1,y:11}, {x:2,y:11}, {x:3,y:11}, {x:4,y:11}, {x:6,y:11}, {x:7,y:11}, {x:8,y:11}, {x:9,y:11}, {x:10,y:11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 70,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
},
level112: {
  player: { x: 5, y: 7 },
  npc: { 
    x: 5, 
    y: 4, 
    movements: {
      experienced1: { path: ["down", "down", "down", "down", "down", "down", "up", "right", "right", "right", "right"], goal: 2 },
      experienced2: { path: ["down", "down", "down", "down", "down", "down", "up", "right", "right", "right", "right"], goal: 2 },
      novice1: { path: ["down", "down", "down", "down", "down", "down", "up", "right", "right", "right", "right"], goal: 2 },
      novice2: { path: ["down", "down", "down", "down", "down", "down", "up", "right", "right", "right", "right"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 2, y: 7, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 3, y: 7, requiredItems: ["redAmulet"], color: 'text-red-500' },
    { x: 8, y: 9, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 7, y: 1, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 1, y: 7, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 9, y: 9, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 1, y: 1, content: 'nothing', color: 'text-blue-500' },
    { x: 1, y: 5, content: 'nothing', color: 'text-blue-500' },
    { x: 9, y: 5, content: 'nothing', color: 'text-blue-500' },
    { x: 1, y: 11, content: 'redAmulet', color: 'text-red-500' },
    { x: 5, y: 11, content: 'blueAmulet', color: 'text-blue-500' },
  ],
  blocks: [{x:0,y:0}, {x:1,y:0}, {x:2,y:0}, {x:3,y:0}, {x:4,y:0}, {x:5,y:0}, {x:6,y:0}, {x:7,y:0}, {x:8,y:0}, {x:9,y:0}, {x:10,y:0}, {x:0,y:1}, {x:2,y:1}, {x:3,y:1}, {x:4,y:1}, {x:5,y:1}, {x:6,y:1}, {x:10,y:1}, {x:0,y:2}, {x:2,y:2}, {x:3,y:2}, {x:4,y:2}, {x:5,y:2}, {x:6,y:2}, {x:7,y:2}, {x:8,y:2}, {x:10,y:2}, {x:0,y:3}, {x:10,y:3}, {x:0,y:4}, {x:1,y:4}, {x:2,y:4}, {x:3,y:4}, {x:4,y:4}, {x:6,y:4}, {x:7,y:4}, {x:8,y:4}, {x:9,y:4}, {x:10,y:4}, {x:0,y:5}, {x:10,y:5}, {x:0,y:6}, {x:1,y:6}, {x:2,y:6}, {x:3,y:6}, {x:4,y:6}, {x:6,y:6}, {x:7,y:6}, {x:8,y:6}, {x:9,y:6}, {x:10,y:6}, {x:0,y:7}, {x:6,y:7}, {x:7,y:7}, {x:8,y:7}, {x:9,y:7}, {x:10,y:7}, {x:0,y:8}, {x:1,y:8}, {x:2,y:8}, {x:3,y:8}, {x:4,y:8}, {x:6,y:8}, {x:7,y:8}, {x:8,y:8}, {x:9,y:8}, {x:10,y:8}, {x:0,y:9}, {x:10,y:9}, {x:0,y:10}, {x:2,y:10}, {x:3,y:10}, {x:4,y:10}, {x:6,y:10}, {x:7,y:10}, {x:8,y:10}, {x:9,y:10}, {x:10,y:10}, {x:0,y:11}, {x:2,y:11}, {x:3,y:11}, {x:4,y:11}, {x:6,y:11}, {x:7,y:11}, {x:8,y:11}, {x:9,y:11}, {x:10,y:11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 70,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
},
level211: {
  player: { x: 2, y: 7 },
  npc: { 
    x: 2, 
    y: 5, 
    movements: {
      experienced1: { path: ["up", "up", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "down", "down"], goal: 1 },
      experienced2: { path: ["left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"], goal: 2 },
      novice1: { path: ["up"], goal: 1 },
      novice2: { path: ["left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 9, y: 9, requiredItems: ["redAmulet"], color: 'text-red-500' },
    { x: 0, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 4, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 10, y: 9, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 0, y: 11, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 4, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 0, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 0, y: 5, content: 'redAmulet', color: 'text-red-500' },
  ],
  blocks: [{x:1,y:0}, {x:2,y:0}, {x:4,y:0}, {x:5,y:0}, {x:7,y:0}, {x:8,y:0}, {x:9,y:0}, {x:1,y:1}, {x:2,y:1}, {x:4,y:1}, {x:5,y:1}, {x:7,y:1}, {x:8,y:1}, {x:9,y:1}, {x:1,y:2}, {x:2,y:2}, {x:4,y:2}, {x:5,y:2}, {x:7,y:2}, {x:8,y:2}, {x:9,y:2}, {x:0,y:4}, {x:1,y:4}, {x:3,y:4}, {x:4,y:4}, {x:5,y:4}, {x:6,y:4}, {x:7,y:4}, {x:8,y:4}, {x:9,y:4}, {x:3,y:5}, {x:4,y:5}, {x:5,y:5}, {x:6,y:5}, {x:7,y:5}, {x:8,y:5}, {x:9,y:5}, {x:0,y:6}, {x:1,y:6}, {x:3,y:6}, {x:4,y:6}, {x:5,y:6}, {x:6,y:6}, {x:7,y:6}, {x:8,y:6}, {x:9,y:6}, {x:0,y:7}, {x:1,y:7}, {x:3,y:7}, {x:4,y:7}, {x:5,y:7}, {x:6,y:7}, {x:7,y:7}, {x:8,y:7}, {x:9,y:7}, {x:10,y:7}, {x:0,y:8}, {x:1,y:8}, {x:3,y:8}, {x:4,y:8}, {x:5,y:8}, {x:6,y:8}, {x:7,y:8}, {x:8,y:8}, {x:9,y:8}, {x:10,y:8}, {x:1,y:10}, {x:2,y:10}, {x:3,y:10}, {x:5,y:10}, {x:6,y:10}, {x:7,y:10}, {x:8,y:10}, {x:9,y:10}, {x:10,y:10}, {x:1,y:11}, {x:2,y:11}, {x:3,y:11}, {x:5,y:11}, {x:6,y:11}, {x:7,y:11}, {x:8,y:11}, {x:9,y:11}, {x:10,y:11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 75,
  goal: {
    type: 'C',
    description: 'Find and obtain Treasure C'
  }
},
level221: {
  player: { x: 0, y: 9 },
  npc: { 
    x: 4, 
    y: 9, 
    movements: {
      experienced1: { path: ["up", "up", "up", "up", "up", "up", "left", "left", "left", "left", "up", "up", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "up", "up"], goal: 1 },
      experienced2: { path: ['down', 'down', 'right', 'right', 'right', 'right', 'right', 'right'], goal: 2 },
      novice1: { path: ["up"], goal: 1 },
      novice2: { path: ['down', 'down', 'right', 'right', 'right', 'right', 'right'], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 0, y: 8, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 0, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 0, y: 7, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 0, y: 11, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 10, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 0, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 10, y: 9, content: 'nothing', color: 'text-blue-500' },
  ],
  blocks: [{ x: 1, y: 0}, { x: 2, y: 0}, { x: 3, y: 0}, { x: 4, y: 0}, { x: 5, y: 0}, { x: 6, y: 0}, { x: 7, y: 0}, { x: 8, y: 0}, { x: 9, y: 0}, { x: 1, y: 1}, { x: 2, y: 1}, { x: 3, y: 1}, { x: 4, y: 1}, { x: 5, y: 1}, { x: 6, y: 1}, { x: 7, y: 1}, { x: 8, y: 1}, { x: 9, y: 1}, { x: 1, y: 2}, { x: 2, y: 2}, { x: 3, y: 2}, { x: 4, y: 2}, { x: 5, y: 2}, { x: 6, y: 2}, { x: 7, y: 2}, { x: 8, y: 2}, { x: 9, y: 2}, { x: 0, y: 4}, { x: 1, y: 4}, { x: 2, y: 4}, { x: 3, y: 4}, { x: 5, y: 4}, { x: 6, y: 4}, { x: 7, y: 4}, { x: 8, y: 4}, { x: 9, y: 4}, { x: 10, y: 4}, { x: 0, y: 5}, { x: 1, y: 5}, { x: 2, y: 5}, { x: 3, y: 5}, { x: 0, y: 6}, { x: 1, y: 6}, { x: 2, y: 6}, { x: 3, y: 6}, { x: 5, y: 6}, { x: 6, y: 6}, { x: 7, y: 6}, { x: 8, y: 6}, { x: 9, y: 6}, { x: 10, y: 6}, { x: 1, y: 7}, { x: 2, y: 7}, { x: 3, y: 7}, { x: 1, y: 8}, { x: 2, y: 8}, { x: 3, y: 8}, { x: 5, y: 8}, { x: 6, y: 8}, { x: 7, y: 8}, { x: 8, y: 8}, { x: 9, y: 8}, { x: 10, y: 8}, { x: 1, y: 10}, { x: 2, y: 10}, { x: 3, y: 10}, { x: 5, y: 10}, { x: 6, y: 10}, { x: 7, y: 10}, { x: 8, y: 10}, { x: 9, y: 10}, { x: 10, y: 10}, { x: 1, y: 11}, { x: 2, y: 11}, { x: 3, y: 11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 85,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
},
level311: {
  player: { x: 2, y: 7 },
  npc: { 
    x: 2, 
    y: 5, 
    movements: {
      experienced1: { path: ["up", "up", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "down", "down"], goal: 1 },
      experienced2: { path: ["left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"], goal: 2 },
      novice1: { path: ["up"], goal: 1 },
      novice2: { path: ["left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 9, y: 9, requiredItems: ["redAmulet"], color: 'text-red-500' },
    { x: 0, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 4, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 10, y: 9, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 0, y: 11, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 4, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 0, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 3, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 0, y: 5, content: 'redAmulet', color: 'text-red-500' },
  ],
  blocks: [{x:1,y:0}, {x:2,y:0}, {x:4,y:0}, {x:5,y:0}, {x:7,y:0}, {x:8,y:0}, {x:9,y:0}, {x:1,y:1}, {x:2,y:1}, {x:4,y:1}, {x:5,y:1}, {x:7,y:1}, {x:8,y:1}, {x:9,y:1}, {x:1,y:2}, {x:2,y:2}, {x:4,y:2}, {x:5,y:2}, {x:7,y:2}, {x:8,y:2}, {x:9,y:2}, {x:0,y:4}, {x:1,y:4}, {x:3,y:4}, {x:4,y:4}, {x:5,y:4}, {x:6,y:4}, {x:7,y:4}, {x:8,y:4}, {x:9,y:4}, {x:3,y:5}, {x:4,y:5}, {x:5,y:5}, {x:6,y:5}, {x:7,y:5}, {x:8,y:5}, {x:9,y:5}, {x:0,y:6}, {x:1,y:6}, {x:3,y:6}, {x:4,y:6}, {x:5,y:6}, {x:6,y:6}, {x:7,y:6}, {x:8,y:6}, {x:9,y:6}, {x:0,y:7}, {x:1,y:7}, {x:3,y:7}, {x:4,y:7}, {x:5,y:7}, {x:6,y:7}, {x:7,y:7}, {x:8,y:7}, {x:9,y:7}, {x:10,y:7}, {x:0,y:8}, {x:1,y:8}, {x:3,y:8}, {x:4,y:8}, {x:5,y:8}, {x:6,y:8}, {x:7,y:8}, {x:8,y:8}, {x:9,y:8}, {x:10,y:8}, {x:1,y:10}, {x:2,y:10}, {x:3,y:10}, {x:5,y:10}, {x:6,y:10}, {x:7,y:10}, {x:8,y:10}, {x:9,y:10}, {x:10,y:10}, {x:1,y:11}, {x:2,y:11}, {x:3,y:11}, {x:5,y:11}, {x:6,y:11}, {x:7,y:11}, {x:8,y:11}, {x:9,y:11}, {x:10,y:11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 75,
  goal: {
    type: 'C',
    description: 'Find and obtain Treasure C'
  }
},
level321: {
  player: { x: 0, y: 9 },
  npc: { 
    x: 4, 
    y: 9, 
    movements: {
      experienced1: { path: ["up", "up", "up", "up", "up", "up", "left", "left", "left", "left", "up", "up", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "up", "up"], goal: 1 },
      experienced2: { path: ['down', 'down', 'right', 'right', 'right', 'right', 'right', 'right'], goal: 2 },
      novice1: { path: ["up"], goal: 1 },
      novice2: { path: ['down', 'down', 'right', 'right', 'right', 'right', 'right', 'right'], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 0, y: 8, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 0, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 0, y: 7, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 0, y: 11, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 10, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 0, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 10, y: 7, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 9, content: 'nothing', color: 'text-blue-500' },
  ],
  blocks: [{ x: 1, y: 0}, { x: 2, y: 0}, { x: 3, y: 0}, { x: 4, y: 0}, { x: 5, y: 0}, { x: 6, y: 0}, { x: 7, y: 0}, { x: 8, y: 0}, { x: 9, y: 0}, { x: 1, y: 1}, { x: 2, y: 1}, { x: 3, y: 1}, { x: 4, y: 1}, { x: 5, y: 1}, { x: 6, y: 1}, { x: 7, y: 1}, { x: 8, y: 1}, { x: 9, y: 1}, { x: 1, y: 2}, { x: 2, y: 2}, { x: 3, y: 2}, { x: 4, y: 2}, { x: 5, y: 2}, { x: 6, y: 2}, { x: 7, y: 2}, { x: 8, y: 2}, { x: 9, y: 2}, { x: 0, y: 4}, { x: 1, y: 4}, { x: 2, y: 4}, { x: 3, y: 4}, { x: 5, y: 4}, { x: 6, y: 4}, { x: 7, y: 4}, { x: 8, y: 4}, { x: 9, y: 4}, { x: 10, y: 4}, { x: 0, y: 5}, { x: 1, y: 5}, { x: 2, y: 5}, { x: 3, y: 5}, { x: 0, y: 6}, { x: 1, y: 6}, { x: 2, y: 6}, { x: 3, y: 6}, { x: 5, y: 6}, { x: 6, y: 6}, { x: 7, y: 6}, { x: 8, y: 6}, { x: 9, y: 6}, { x: 10, y: 6}, { x: 1, y: 7}, { x: 2, y: 7}, { x: 3, y: 7}, { x: 1, y: 8}, { x: 2, y: 8}, { x: 3, y: 8}, { x: 5, y: 8}, { x: 6, y: 8}, { x: 7, y: 8}, { x: 8, y: 8}, { x: 9, y: 8}, { x: 10, y: 8}, { x: 1, y: 10}, { x: 2, y: 10}, { x: 3, y: 10}, { x: 5, y: 10}, { x: 6, y: 10}, { x: 7, y: 10}, { x: 8, y: 10}, { x: 9, y: 10}, { x: 10, y: 10}, { x: 1, y: 11}, { x: 2, y: 11}, { x: 3, y: 11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 85,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
},
level331: {
  player: { x: 3, y: 11 },
  npc: { 
    x: 3, 
    y: 2, 
    movements: {
      experienced1: { path: ["left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up"], goal: 1 },
      experienced2: { path: ["down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down"], goal: 2 },
      novice1: { path: ["right", "up", "down", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up"], goal: 1 },
      novice2: { path: ["down", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 10, y: 1, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 0, y: 8, requiredItems: ["redAmulet"], color: 'text-red-500' },
    { x: 9, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 10, y: 0, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 10, y: 10, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 0, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 4, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 0, y: 2, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 10, y: 7, content: 'redAmulet', color: 'text-red-500' },
  ],
  blocks: [{ x: 0, y: 0}, { x: 1, y: 0}, { x: 2, y: 0}, { x: 3, y: 0}, { x: 5, y: 0}, { x: 6, y: 0}, { x: 7, y: 0}, { x: 8, y: 0}, { x: 9, y: 0}, { x: 0, y: 1}, { x: 1, y: 1}, { x: 2, y: 1}, { x: 3, y: 1}, { x: 5, y: 1}, { x: 6, y: 1}, { x: 7, y: 1}, { x: 8, y: 1}, { x: 9, y: 1}, { x: 0, y: 3}, { x: 1, y: 3}, { x: 2, y: 3}, { x: 4, y: 3}, { x: 5, y: 3}, { x: 6, y: 3}, { x: 7, y: 3}, { x: 8, y: 3}, { x: 9, y: 3}, { x: 10, y: 3}, { x: 0, y: 4}, { x: 1, y: 4}, { x: 2, y: 4}, { x: 4, y: 4}, { x: 5, y: 4}, { x: 6, y: 4}, { x: 7, y: 4}, { x: 8, y: 4}, { x: 9, y: 4}, { x: 10, y: 4}, { x: 0, y: 5}, { x: 1, y: 5}, { x: 2, y: 5}, { x: 4, y: 5}, { x: 5, y: 5}, { x: 6, y: 5}, { x: 7, y: 5}, { x: 8, y: 5}, { x: 9, y: 5}, { x: 10, y: 5}, { x: 0, y: 6}, { x: 1, y: 6}, { x: 2, y: 6}, { x: 4, y: 6}, { x: 5, y: 6}, { x: 6, y: 6}, { x: 7, y: 6}, { x: 8, y: 6}, { x: 9, y: 6}, { x: 10, y: 6}, { x: 1, y: 8}, { x: 2, y: 8}, { x: 4, y: 8}, { x: 5, y: 8}, { x: 6, y: 8}, { x: 8, y: 8}, { x: 9, y: 8}, { x: 10, y: 8}, { x: 1, y: 9}, { x: 2, y: 9}, { x: 4, y: 9}, { x: 5, y: 9}, { x: 6, y: 9}, { x: 8, y: 9}, { x: 9, y: 9}, { x: 10, y: 9}, { x: 1, y: 10}, { x: 2, y: 10}, { x: 4, y: 10}, { x: 5, y: 10}, { x: 6, y: 10}, { x: 1, y: 11}, { x: 2, y: 11}, { x: 4, y: 11}, { x: 5, y: 11}, { x: 6, y: 11}, { x: 7, y: 11}, { x: 8, y: 11}, { x: 9, y: 11}, { x: 10, y: 11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 50,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
},
level332: {
  player: { x: 3, y: 11 },
  npc: { 
    x: 3, 
    y: 5, 
    movements: {
      experienced1: { path: ["up", "up", "up", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up"], goal: 1 },
      experienced2: { path: ["down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down"], goal: 2 },
      novice1: { path: ["up", "up", "up", "right", "up", "down", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up"], goal: 1 },
      novice2: { path: ["down", "down", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down"], goal: 2 },
    }, 
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 10, y: 1, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 0, y: 8, requiredItems: ["redAmulet"], color: 'text-red-500' },
    { x: 9, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 10, y: 0, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 10, y: 10, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 0, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 4, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 0, y: 2, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 10, y: 7, content: 'redAmulet', color: 'text-red-500' },
  ],
  blocks: [{ x: 0, y: 0}, { x: 1, y: 0}, { x: 2, y: 0}, { x: 3, y: 0}, { x: 5, y: 0}, { x: 6, y: 0}, { x: 7, y: 0}, { x: 8, y: 0}, { x: 9, y: 0}, { x: 0, y: 1}, { x: 1, y: 1}, { x: 2, y: 1}, { x: 3, y: 1}, { x: 5, y: 1}, { x: 6, y: 1}, { x: 7, y: 1}, { x: 8, y: 1}, { x: 9, y: 1}, { x: 0, y: 3}, { x: 1, y: 3}, { x: 2, y: 3}, { x: 4, y: 3}, { x: 5, y: 3}, { x: 6, y: 3}, { x: 7, y: 3}, { x: 8, y: 3}, { x: 9, y: 3}, { x: 10, y: 3}, { x: 0, y: 4}, { x: 1, y: 4}, { x: 2, y: 4}, { x: 4, y: 4}, { x: 5, y: 4}, { x: 6, y: 4}, { x: 7, y: 4}, { x: 8, y: 4}, { x: 9, y: 4}, { x: 10, y: 4}, { x: 0, y: 5}, { x: 1, y: 5}, { x: 2, y: 5}, { x: 4, y: 5}, { x: 5, y: 5}, { x: 6, y: 5}, { x: 7, y: 5}, { x: 8, y: 5}, { x: 9, y: 5}, { x: 10, y: 5}, { x: 0, y: 6}, { x: 1, y: 6}, { x: 2, y: 6}, { x: 4, y: 6}, { x: 5, y: 6}, { x: 6, y: 6}, { x: 7, y: 6}, { x: 8, y: 6}, { x: 9, y: 6}, { x: 10, y: 6}, { x: 1, y: 8}, { x: 2, y: 8}, { x: 4, y: 8}, { x: 5, y: 8}, { x: 6, y: 8}, { x: 8, y: 8}, { x: 9, y: 8}, { x: 10, y: 8}, { x: 1, y: 9}, { x: 2, y: 9}, { x: 4, y: 9}, { x: 5, y: 9}, { x: 6, y: 9}, { x: 8, y: 9}, { x: 9, y: 9}, { x: 10, y: 9}, { x: 1, y: 10}, { x: 2, y: 10}, { x: 4, y: 10}, { x: 5, y: 10}, { x: 6, y: 10}, { x: 1, y: 11}, { x: 2, y: 11}, { x: 4, y: 11}, { x: 5, y: 11}, { x: 6, y: 11}, { x: 7, y: 11}, { x: 8, y: 11}, { x: 9, y: 11}, { x: 10, y: 11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 40,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
},
level341: {
  player: { x: 10, y: 8 },
  npc: { 
    x: 6, 
    y: 3, 
    movements: {
      experienced1: { path: ["left", "left", "left", "left", "left", "left", "down", "up", "right", "right", "right", "down", "down", "down", "down", "down", "right", "right", "down", "down", "down"], goal: 1 },
      experienced2: { path: ["left", "up", "up", "down", "down", "right", "right", "right", "right", "up", "up", "up"], goal: 2 },
      novice1: { path: ["left", "left", "left", "left", "left", "left", "up", "down", "down", "up", "right", "right", "right", "down", "down", "down", "down", "down", "right", "right", "down", "down", "down"], goal: 1 },
      novice2: { path: ["left", "up", "up", "down", "down", "right", "right", "right", "right", "up", "up", "up"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 9, y: 1, requiredItems: ["redAmulet"], color: 'text-red-500' },
    { x: 5, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 9, y: 0, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 5, y: 11, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 9, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 5, y: 0, content: 'redAmulet', color: 'text-red-500' },
    { x: 0, y: 1, content: 'nothing', color: 'text-blue-500' },
    { x: 0, y: 5, content: 'blueAmulet', color: 'text-blue-500' },
  ],
  blocks: [{ x: 0, y: 0}, { x: 1, y: 0}, { x: 2, y: 0}, { x: 3, y: 0}, { x: 4, y: 0}, { x: 6, y: 0}, { x: 7, y: 0}, { x: 8, y: 0}, { x: 10, y: 0}, { x: 1, y: 1}, { x: 2, y: 1}, { x: 3, y: 1}, { x: 4, y: 1}, { x: 6, y: 1}, { x: 7, y: 1}, { x: 8, y: 1}, { x: 10, y: 1}, { x: 1, y: 2}, { x: 2, y: 2}, { x: 3, y: 2}, { x: 4, y: 2}, { x: 6, y: 2}, { x: 7, y: 2}, { x: 8, y: 2}, { x: 10, y: 2}, { x: 10, y: 3}, { x: 1, y: 4}, { x: 2, y: 4}, { x: 4, y: 4}, { x: 5, y: 4}, { x: 6, y: 4}, { x: 7, y: 4}, { x: 8, y: 4}, { x: 9, y: 4}, { x: 10, y: 4}, { x: 1, y: 5}, { x: 2, y: 5}, { x: 4, y: 5}, { x: 5, y: 5}, { x: 6, y: 5}, { x: 7, y: 5}, { x: 8, y: 5}, { x: 9, y: 5}, { x: 10, y: 5}, { x: 0, y: 6}, { x: 1, y: 6}, { x: 2, y: 6}, { x: 4, y: 6}, { x: 5, y: 6}, { x: 6, y: 6}, { x: 7, y: 6}, { x: 8, y: 6}, { x: 9, y: 6}, { x: 10, y: 6}, { x: 0, y: 7}, { x: 1, y: 7}, { x: 2, y: 7}, { x: 4, y: 7}, { x: 5, y: 7}, { x: 6, y: 7}, { x: 7, y: 7}, { x: 8, y: 7}, { x: 9, y: 7}, { x: 10, y: 7}, { x: 0, y: 9}, { x: 1, y: 9}, { x: 2, y: 9}, { x: 3, y: 9}, { x: 4, y: 9}, { x: 6, y: 9}, { x: 7, y: 9}, { x: 8, y: 9}, { x: 10, y: 9}, { x: 0, y: 10}, { x: 1, y: 10}, { x: 2, y: 10}, { x: 3, y: 10}, { x: 4, y: 10}, { x: 6, y: 10}, { x: 7, y: 10}, { x: 8, y: 10}, { x: 10, y: 10}, { x: 0, y: 11}, { x: 1, y: 11}, { x: 2, y: 11}, { x: 3, y: 11}, { x: 4, y: 11}, { x: 6, y: 11}, { x: 7, y: 11}, { x: 8, y: 11}, { x: 10, y: 11}],
  openedBarriers: [],
  stepLimit: 70,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
},
level342: {
  player: { x: 10, y: 8 },
  npc: { 
    x: 2, 
    y: 3, 
    movements: {
      experienced1: { path: ["left", "left", "down", "up", "right", "right", "right", "right", "right", "up", "up", "down", "down", "left", "left", "down", "down", "down", "down", "down", "right", "right", "down", "down", "down"], goal: 1 },
      experienced2: { path: ["right", "right", "right", "up", "up", "down", "down", "right", "right", "right", "right", "up", "up", "up"], goal: 2 },
      novice1: { path: ["left", "left", "up", "down", "down", "up", "right", "right", "right", "right", "right", "up", "up", "down", "down", "left", "left", "down", "down", "down", "down", "down", "right", "right", "down", "down", "down", "down"], goal: 1 },
      novice2: { path: ["right", "right", "right", "up", "up", "down", "down", "right", "right", "right", "right", "up", "up", "up"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 9, y: 1, requiredItems: ["redAmulet"], color: 'text-red-500' },
    { x: 5, y: 9, requiredItems: ["redAmulet"], color: 'text-red-500' },
    { x: 5, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 9, y: 0, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 5, y: 11, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 9, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 5, y: 0, content: 'redAmulet', color: 'text-red-500' },
    { x: 0, y: 1, content: 'nothing', color: 'text-blue-500' },
    { x: 0, y: 5, content: 'blueAmulet', color: 'text-blue-500' },
  ],
  blocks: [{x:0,y:0}, {x:1,y:0}, {x:2,y:0}, {x:3,y:0}, {x:4,y:0}, {x:6,y:0}, {x:7,y:0}, {x:8,y:0}, {x:10,y:0}, {x:1,y:1}, {x:2,y:1}, {x:3,y:1}, {x:4,y:1}, {x:6,y:1}, {x:7,y:1}, {x:8,y:1}, {x:10,y:1}, {x:1,y:2}, {x:2,y:2}, {x:3,y:2}, {x:4,y:2}, {x:6,y:2}, {x:7,y:2}, {x:8,y:2}, {x:10,y:2}, {x:10,y:3}, {x:1,y:4}, {x:2,y:4}, {x:4,y:4}, {x:5,y:4}, {x:6,y:4}, {x:8,y:4}, {x:9,y:4}, {x:10,y:4}, {x:1,y:5}, {x:2,y:5}, {x:4,y:5}, {x:5,y:5}, {x:6,y:5}, {x:8,y:5}, {x:9,y:5}, {x:10,y:5}, {x:0,y:6}, {x:1,y:6}, {x:2,y:6}, {x:4,y:6}, {x:5,y:6}, {x:6,y:6}, {x:8,y:6}, {x:9,y:6}, {x:10,y:6}, {x:0,y:7}, {x:1,y:7}, {x:2,y:7}, {x:4,y:7}, {x:5,y:7}, {x:6,y:7}, {x:8,y:7}, {x:9,y:7}, {x:10,y:7}, {x:0,y:9}, {x:1,y:9}, {x:2,y:9}, {x:3,y:9}, {x:4,y:9}, {x:6,y:9}, {x:7,y:9}, {x:8,y:9}, {x:10,y:9}, {x:0,y:10}, {x:1,y:10}, {x:2,y:10}, {x:3,y:10}, {x:4,y:10}, {x:6,y:10}, {x:7,y:10}, {x:8,y:10}, {x:10,y:10}, {x:0,y:11}, {x:1,y:11}, {x:2,y:11}, {x:3,y:11}, {x:4,y:11}, {x:6,y:11}, {x:7,y:11}, {x:8,y:11}, {x:10,y:11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 80,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
},
level351: {
  player: { x: 5, y: 7 },
  npc: { 
    x: 4, 
    y: 2, 
    movements: {
      experienced1: { path: ["right", "right", "right", "right", "right", "right", "up", "down", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right"], goal: 1 },
      experienced2: { path: ["left", "left", "left", "left"], goal: 2 },
      novice1: { path: ["up"], goal: 1 },
      novice2: { path: ["left", "left", "left", "left"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 2, y: 7, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 8, y: 7, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 0, y: 2, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 0, y: 7, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 10, y: 7, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 5, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 7, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
  ],
  blocks: [{x:0,y:0}, {x:1,y:0}, {x:2,y:0}, {x:3,y:0}, {x:4,y:0}, {x:6,y:0}, {x:8,y:0}, {x:9,y:0}, {x:0,y:1}, {x:1,y:1}, {x:2,y:1}, {x:3,y:1}, {x:4,y:1}, {x:6,y:1}, {x:8,y:1}, {x:9,y:1}, {x:0,y:3}, {x:1,y:3}, {x:2,y:3}, {x:3,y:3}, {x:4,y:3}, {x:6,y:3}, {x:7,y:3}, {x:8,y:3}, {x:9,y:3}, {x:10,y:3}, {x:0,y:4}, {x:1,y:4}, {x:2,y:4}, {x:3,y:4}, {x:4,y:4}, {x:0,y:5}, {x:1,y:5}, {x:2,y:5}, {x:3,y:5}, {x:4,y:5}, {x:6,y:5}, {x:7,y:5}, {x:8,y:5}, {x:9,y:5}, {x:10,y:5}, {x:0,y:6}, {x:1,y:6}, {x:2,y:6}, {x:3,y:6}, {x:4,y:6}, {x:6,y:6}, {x:7,y:6}, {x:8,y:6}, {x:9,y:6}, {x:10,y:6}, {x:0,y:8}, {x:1,y:8}, {x:2,y:8}, {x:3,y:8}, {x:4,y:8}, {x:6,y:8}, {x:7,y:8}, {x:8,y:8}, {x:9,y:8}, {x:10,y:8}, {x:0,y:9}, {x:1,y:9}, {x:2,y:9}, {x:3,y:9}, {x:4,y:9}, {x:6,y:9}, {x:7,y:9}, {x:8,y:9}, {x:9,y:9}, {x:10,y:9}, {x:0,y:11}, {x:1,y:11}, {x:2,y:11}, {x:3,y:11}, {x:4,y:11}, {x:5,y:11}, {x:6,y:11}, {x:7,y:11}, {x:8,y:11}, {x:9,y:11}, {x:10,y:11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 55,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
},
level352: {
  player: { x: 5, y: 7 },
    npc: { 
      x: 5, 
      y: 4, 
    movements: {
      experienced1: { path: ["up", "up", "right", "right", "right", "right", "right", "up", "down", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right"], goal: 1 },
      experienced2: { path: ["up", "up", "left", "left", "left", "left", "left"], goal: 2 },
      novice1: { path: ["up"], goal: 1 },
      novice2: { path: ["up", "up", "left", "left", "left", "left", "left"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 2, y: 7, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 8, y: 7, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 0, y: 2, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 0, y: 7, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 10, y: 7, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 5, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 7, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
  ],
  blocks: [{x:0,y:0}, {x:1,y:0}, {x:2,y:0}, {x:3,y:0}, {x:4,y:0}, {x:6,y:0}, {x:8,y:0}, {x:9,y:0}, {x:0,y:1}, {x:1,y:1}, {x:2,y:1}, {x:3,y:1}, {x:4,y:1}, {x:6,y:1}, {x:8,y:1}, {x:9,y:1}, {x:0,y:3}, {x:1,y:3}, {x:2,y:3}, {x:3,y:3}, {x:4,y:3}, {x:6,y:3}, {x:7,y:3}, {x:8,y:3}, {x:9,y:3}, {x:10,y:3}, {x:0,y:4}, {x:1,y:4}, {x:2,y:4}, {x:3,y:4}, {x:4,y:4}, {x:0,y:5}, {x:1,y:5}, {x:2,y:5}, {x:3,y:5}, {x:4,y:5}, {x:6,y:5}, {x:7,y:5}, {x:8,y:5}, {x:9,y:5}, {x:10,y:5}, {x:0,y:6}, {x:1,y:6}, {x:2,y:6}, {x:3,y:6}, {x:4,y:6}, {x:6,y:6}, {x:7,y:6}, {x:8,y:6}, {x:9,y:6}, {x:10,y:6}, {x:0,y:8}, {x:1,y:8}, {x:2,y:8}, {x:3,y:8}, {x:4,y:8}, {x:6,y:8}, {x:7,y:8}, {x:8,y:8}, {x:9,y:8}, {x:10,y:8}, {x:0,y:9}, {x:1,y:9}, {x:2,y:9}, {x:3,y:9}, {x:4,y:9}, {x:6,y:9}, {x:7,y:9}, {x:8,y:9}, {x:9,y:9}, {x:10,y:9}, {x:0,y:11}, {x:1,y:11}, {x:2,y:11}, {x:3,y:11}, {x:4,y:11}, {x:5,y:11}, {x:6,y:11}, {x:7,y:11}, {x:8,y:11}, {x:9,y:11}, {x:10,y:11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 65,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
},
level361: {
  player: { x: 4, y: 7 },
  npc: { 
    x: 4, 
    y: 5, 
    movements: {
      experienced1: { path: ["up", "up", "right", "right", "right", "right", "right", "right", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "down", "down", "right", "right", "right"], goal: 1 },
      experienced2: { path: ["down", "down", "down", "down", "down", "down"], goal: 2 },
      novice1: { path: ["up"], goal: 1 },
      novice2: { path: ["down", "down", "down", "down", "down", "down", "down"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 0, y: 10, requiredItems: ["redAmulet"], color: 'text-red-500' },
    { x: 8, y: 11, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 0, y: 11, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 4, y: 11, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 10, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 0, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 10, y: 5, content: 'redAmulet', color: 'text-red-500' },
    { x: 0, y: 7, content: 'nothing', color: 'text-blue-500' },
  ],
  blocks: [{x:1,y:0}, {x:2,y:0}, {x:3,y:0}, {x:4,y:0}, {x:5,y:0}, {x:6,y:0}, {x:7,y:0}, {x:8,y:0}, {x:9,y:0}, {x:1,y:1}, {x:2,y:1}, {x:3,y:1}, {x:4,y:1}, {x:5,y:1}, {x:6,y:1}, {x:7,y:1}, {x:8,y:1}, {x:9,y:1}, {x:1,y:2}, {x:2,y:2}, {x:3,y:2}, {x:4,y:2}, {x:5,y:2}, {x:6,y:2}, {x:7,y:2}, {x:8,y:2}, {x:9,y:2}, {x:0,y:4}, {x:1,y:4}, {x:2,y:4}, {x:3,y:4}, {x:5,y:4}, {x:6,y:4}, {x:7,y:4}, {x:8,y:4}, {x:9,y:4}, {x:10,y:4}, {x:0,y:5}, {x:1,y:5}, {x:2,y:5}, {x:3,y:5}, {x:0,y:6}, {x:1,y:6}, {x:2,y:6}, {x:3,y:6}, {x:5,y:6}, {x:6,y:6}, {x:7,y:6}, {x:8,y:6}, {x:9,y:6}, {x:10,y:6}, {x:5,y:7}, {x:6,y:7}, {x:7,y:7}, {x:8,y:7}, {x:9,y:7}, {x:10,y:7}, {x:0,y:8}, {x:1,y:8}, {x:2,y:8}, {x:3,y:8}, {x:5,y:8}, {x:6,y:8}, {x:7,y:8}, {x:8,y:8}, {x:9,y:8}, {x:10,y:8}, {x:1,y:10}, {x:2,y:10}, {x:3,y:10}, {x:5,y:10}, {x:6,y:10}, {x:8,y:10}, {x:9,y:10}, {x:10,y:10}, {x:1,y:11}, {x:2,y:11}, {x:3,y:11}, {x:5,y:11}, {x:6,y:11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 50,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
},
level362: {
  player: { x: 4, y: 3 },
  npc: { 
    x: 4, 
    y: 7, 
    movements: {
      experienced1: { path: ["up", "up", "up", "up", "right", "right", "right", "right", "right", "right", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "down", "down", "right", "right", "right"], goal: 1 },
      experienced2: { path: ["up", "up", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"], goal: 2 },
      novice1: { path: ["up"], goal: 1 },
      novice2: { path: ["up", "up", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "down", "down", "down", "down", "left", "left", "left", "left", "down", "down"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 0, y: 10, requiredItems: ["redAmulet"], color: 'text-red-500' },
    { x: 8, y: 11, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 0, y: 11, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 4, y: 11, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 10, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 0, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 10, y: 5, content: 'redAmulet', color: 'text-red-500' },
    { x: 0, y: 7, content: 'nothing', color: 'text-blue-500' },
  ],
  blocks: [{x:1,y:0}, {x:2,y:0}, {x:3,y:0}, {x:4,y:0}, {x:5,y:0}, {x:6,y:0}, {x:7,y:0}, {x:8,y:0}, {x:9,y:0}, {x:1,y:1}, {x:2,y:1}, {x:3,y:1}, {x:4,y:1}, {x:5,y:1}, {x:6,y:1}, {x:7,y:1}, {x:8,y:1}, {x:9,y:1}, {x:1,y:2}, {x:2,y:2}, {x:3,y:2}, {x:4,y:2}, {x:5,y:2}, {x:6,y:2}, {x:7,y:2}, {x:8,y:2}, {x:9,y:2}, {x:0,y:4}, {x:1,y:4}, {x:2,y:4}, {x:3,y:4}, {x:5,y:4}, {x:6,y:4}, {x:7,y:4}, {x:8,y:4}, {x:9,y:4}, {x:10,y:4}, {x:0,y:5}, {x:1,y:5}, {x:2,y:5}, {x:3,y:5}, {x:0,y:6}, {x:1,y:6}, {x:2,y:6}, {x:3,y:6}, {x:5,y:6}, {x:6,y:6}, {x:7,y:6}, {x:8,y:6}, {x:9,y:6}, {x:10,y:6}, {x:5,y:7}, {x:6,y:7}, {x:7,y:7}, {x:8,y:7}, {x:9,y:7}, {x:10,y:7}, {x:0,y:8}, {x:1,y:8}, {x:2,y:8}, {x:3,y:8}, {x:5,y:8}, {x:6,y:8}, {x:7,y:8}, {x:8,y:8}, {x:9,y:8}, {x:10,y:8}, {x:1,y:10}, {x:2,y:10}, {x:3,y:10}, {x:5,y:10}, {x:6,y:10}, {x:8,y:10}, {x:9,y:10}, {x:10,y:10}, {x:1,y:11}, {x:2,y:11}, {x:3,y:11}, {x:5,y:11}, {x:6,y:11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 20,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
},
level411: {
  player: { x: 2, y: 7 },
  npc: { 
    x: 2, 
    y: 5, 
    movements: {
      experienced1: { path: ["up", "up", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "down", "down"], goal: 1 },
      experienced2: { path: ["left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"], goal: 2 },
      novice1: { path: ["up"], goal: 1 },
      novice2: { path: ["left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 9, y: 9, requiredItems: ["redAmulet"], color: 'text-red-500' },
    { x: 0, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 4, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 10, y: 9, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 0, y: 11, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 4, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 0, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 3, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 6, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 0, y: 5, content: 'redAmulet', color: 'text-red-500' },
  ],
  blocks: [{x:1,y:0}, {x:2,y:0}, {x:4,y:0}, {x:5,y:0}, {x:7,y:0}, {x:8,y:0}, {x:9,y:0}, {x:1,y:1}, {x:2,y:1}, {x:4,y:1}, {x:5,y:1}, {x:7,y:1}, {x:8,y:1}, {x:9,y:1}, {x:1,y:2}, {x:2,y:2}, {x:4,y:2}, {x:5,y:2}, {x:7,y:2}, {x:8,y:2}, {x:9,y:2}, {x:0,y:4}, {x:1,y:4}, {x:3,y:4}, {x:4,y:4}, {x:5,y:4}, {x:6,y:4}, {x:7,y:4}, {x:8,y:4}, {x:9,y:4}, {x:3,y:5}, {x:4,y:5}, {x:5,y:5}, {x:6,y:5}, {x:7,y:5}, {x:8,y:5}, {x:9,y:5}, {x:0,y:6}, {x:1,y:6}, {x:3,y:6}, {x:4,y:6}, {x:5,y:6}, {x:6,y:6}, {x:7,y:6}, {x:8,y:6}, {x:9,y:6}, {x:0,y:7}, {x:1,y:7}, {x:3,y:7}, {x:4,y:7}, {x:5,y:7}, {x:6,y:7}, {x:7,y:7}, {x:8,y:7}, {x:9,y:7}, {x:10,y:7}, {x:0,y:8}, {x:1,y:8}, {x:3,y:8}, {x:4,y:8}, {x:5,y:8}, {x:6,y:8}, {x:7,y:8}, {x:8,y:8}, {x:9,y:8}, {x:10,y:8}, {x:1,y:10}, {x:2,y:10}, {x:3,y:10}, {x:5,y:10}, {x:6,y:10}, {x:7,y:10}, {x:8,y:10}, {x:9,y:10}, {x:10,y:10}, {x:1,y:11}, {x:2,y:11}, {x:3,y:11}, {x:5,y:11}, {x:6,y:11}, {x:7,y:11}, {x:8,y:11}, {x:9,y:11}, {x:10,y:11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 80,
  goal: {
    type: 'C',
    description: 'Find and obtain Treasure C'
  }
},
level421: {
  player: { x: 0, y: 9 },
  npc: { 
    x: 4, 
    y: 9, 
    movements: {
      experienced1: { path: ["up", "up", "up", "up", "up", "up", "left", "left", "left", "left", "up", "up", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "up", "up"], goal: 1 },
      experienced2: { path: ['down', 'down', 'right', 'right', 'right', 'right', 'right', 'right'], goal: 2 },
      novice1: { path: ["up"], goal: 1 },
      novice2: { path: ['down', 'down', 'right', 'right', 'right', 'right', 'right', 'right'], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 0, y: 8, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 0, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 0, y: 7, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 0, y: 11, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 10, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 0, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 10, y: 5, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 7, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 9, content: 'nothing', color: 'text-blue-500' },
  ],
  blocks: [{ x: 1, y: 0}, { x: 2, y: 0}, { x: 3, y: 0}, { x: 4, y: 0}, { x: 5, y: 0}, { x: 6, y: 0}, { x: 7, y: 0}, { x: 8, y: 0}, { x: 9, y: 0}, { x: 1, y: 1}, { x: 2, y: 1}, { x: 3, y: 1}, { x: 4, y: 1}, { x: 5, y: 1}, { x: 6, y: 1}, { x: 7, y: 1}, { x: 8, y: 1}, { x: 9, y: 1}, { x: 1, y: 2}, { x: 2, y: 2}, { x: 3, y: 2}, { x: 4, y: 2}, { x: 5, y: 2}, { x: 6, y: 2}, { x: 7, y: 2}, { x: 8, y: 2}, { x: 9, y: 2}, { x: 0, y: 4}, { x: 1, y: 4}, { x: 2, y: 4}, { x: 3, y: 4}, { x: 5, y: 4}, { x: 6, y: 4}, { x: 7, y: 4}, { x: 8, y: 4}, { x: 9, y: 4}, { x: 10, y: 4}, { x: 0, y: 5}, { x: 1, y: 5}, { x: 2, y: 5}, { x: 3, y: 5}, { x: 0, y: 6}, { x: 1, y: 6}, { x: 2, y: 6}, { x: 3, y: 6}, { x: 5, y: 6}, { x: 6, y: 6}, { x: 7, y: 6}, { x: 8, y: 6}, { x: 9, y: 6}, { x: 10, y: 6}, { x: 1, y: 7}, { x: 2, y: 7}, { x: 3, y: 7}, { x: 1, y: 8}, { x: 2, y: 8}, { x: 3, y: 8}, { x: 5, y: 8}, { x: 6, y: 8}, { x: 7, y: 8}, { x: 8, y: 8}, { x: 9, y: 8}, { x: 10, y: 8}, { x: 1, y: 10}, { x: 2, y: 10}, { x: 3, y: 10}, { x: 5, y: 10}, { x: 6, y: 10}, { x: 7, y: 10}, { x: 8, y: 10}, { x: 9, y: 10}, { x: 10, y: 10}, { x: 1, y: 11}, { x: 2, y: 11}, { x: 3, y: 11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 85,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
},
level431: {
  player: { x: 5, y: 9 },
  npc: { 
    x: 0, 
    y: 5, 
    movements: {
      experienced1: { path: ["up", "up", "right", "right", "right", "right", "right", "up", "up", "down", "down", "left", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "up", "up"], goal: 1 },
      experienced2: { path: ["right", "right", "right", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down"], goal: 2 },
      novice1: { path: ["up"], goal: 1 },
      novice2: { path: ["right", "right", "right", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 10, y: 1, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 0, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 5, y: 10, requiredItems: ["redAmulet"], color: 'text-red-500' },
  ],
  treasurePots: [
    { x: 10, y: 0, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 0, y: 11, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 5, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 0, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 5, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 10, y: 7, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 9, content: 'redAmulet', color: 'text-red-500' },
  ],
  blocks: [{x:1,y:0}, {x:2,y:0}, {x:3,y:0}, {x:4,y:0}, {x:6,y:0}, {x:7,y:0}, {x:8,y:0}, {x:9,y:0}, {x:1,y:1}, {x:2,y:1}, {x:3,y:1}, {x:4,y:1}, {x:6,y:1}, {x:7,y:1}, {x:8,y:1}, {x:9,y:1}, {x:1,y:2}, {x:2,y:2}, {x:3,y:2}, {x:4,y:2}, {x:6,y:2}, {x:7,y:2}, {x:8,y:2}, {x:9,y:2}, {x:6,y:3}, {x:7,y:3}, {x:8,y:3}, {x:9,y:3}, {x:1,y:4}, {x:2,y:4}, {x:3,y:4}, {x:4,y:4}, {x:5,y:4}, {x:6,y:4}, {x:7,y:4}, {x:8,y:4}, {x:9,y:4}, {x:0,y:6}, {x:1,y:6}, {x:2,y:6}, {x:3,y:6}, {x:5,y:6}, {x:6,y:6}, {x:7,y:6}, {x:8,y:6}, {x:9,y:6}, {x:0,y:7}, {x:1,y:7}, {x:2,y:7}, {x:3,y:7}, {x:5,y:7}, {x:6,y:7}, {x:7,y:7}, {x:8,y:7}, {x:9,y:7}, {x:0,y:8}, {x:1,y:8}, {x:2,y:8}, {x:3,y:8}, {x:5,y:8}, {x:6,y:8}, {x:7,y:8}, {x:8,y:8}, {x:9,y:8}, {x:10,y:8}, {x:1,y:10}, {x:2,y:10}, {x:3,y:10}, {x:4,y:10}, {x:6,y:10}, {x:7,y:10}, {x:8,y:10}, {x:9,y:10}, {x:10,y:10}, {x:1,y:11}, {x:2,y:11}, {x:3,y:11}, {x:4,y:11}, {x:6,y:11}, {x:7,y:11}, {x:8,y:11}, {x:9,y:11}, {x:10,y:11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 90,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
},
level432: {
  player: { x: 5, y: 9 },
  npc: { 
    x: 4, 
    y: 7, 
    movements: {
      experienced1: { path: ["up", "up", "left", "left", "left", "left", "up", "up", "right", "right", "right", "right", "right", "up", "up", "down", "down", "left", "left", "left", "left", "left", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up", "up", "up"], goal: 1 },
      experienced2: { path: ["down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down"], goal: 2 },
      novice1: { path: ["up"], goal: 1 },
      novice2: { path: ["down", "down", "right", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 10, y: 1, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 0, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 5, y: 10, requiredItems: ["redAmulet"], color: 'text-red-500' },
  ],
  treasurePots: [
    { x: 10, y: 0, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 0, y: 11, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 5, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 0, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 5, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 10, y: 7, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 9, content: 'redAmulet', color: 'text-red-500' },
  ],
  blocks: [{x:1,y:0}, {x:2,y:0}, {x:3,y:0}, {x:4,y:0}, {x:6,y:0}, {x:7,y:0}, {x:8,y:0}, {x:9,y:0}, {x:1,y:1}, {x:2,y:1}, {x:3,y:1}, {x:4,y:1}, {x:6,y:1}, {x:7,y:1}, {x:8,y:1}, {x:9,y:1}, {x:1,y:2}, {x:2,y:2}, {x:3,y:2}, {x:4,y:2}, {x:6,y:2}, {x:7,y:2}, {x:8,y:2}, {x:9,y:2}, {x:6,y:3}, {x:7,y:3}, {x:8,y:3}, {x:9,y:3}, {x:1,y:4}, {x:2,y:4}, {x:3,y:4}, {x:4,y:4}, {x:5,y:4}, {x:6,y:4}, {x:7,y:4}, {x:8,y:4}, {x:9,y:4}, {x:0,y:6}, {x:1,y:6}, {x:2,y:6}, {x:3,y:6}, {x:5,y:6}, {x:6,y:6}, {x:7,y:6}, {x:8,y:6}, {x:9,y:6}, {x:0,y:7}, {x:1,y:7}, {x:2,y:7}, {x:3,y:7}, {x:5,y:7}, {x:6,y:7}, {x:7,y:7}, {x:8,y:7}, {x:9,y:7}, {x:0,y:8}, {x:1,y:8}, {x:2,y:8}, {x:3,y:8}, {x:5,y:8}, {x:6,y:8}, {x:7,y:8}, {x:8,y:8}, {x:9,y:8}, {x:10,y:8}, {x:1,y:10}, {x:2,y:10}, {x:3,y:10}, {x:4,y:10}, {x:6,y:10}, {x:7,y:10}, {x:8,y:10}, {x:9,y:10}, {x:10,y:10}, {x:1,y:11}, {x:2,y:11}, {x:3,y:11}, {x:4,y:11}, {x:6,y:11}, {x:7,y:11}, {x:8,y:11}, {x:9,y:11}, {x:10,y:11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 95,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
},
level441: {
  player: { x: 5, y: 3 },
  npc: { 
    x: 5, 
    y: 7, 
    movements: {
      experienced1: { path: ["down", "down", "right", "right", "right", "down", "up", "right", "right", "down", "down"], goal: 1 },
      experienced2: { path: ["up", "up", "up", "up", "up", "up", "up"], goal: 2 },
      novice1: { path: ["up"], goal: 1 },
      novice2: { path: ["up", "up", "up", "up", "up", "up", "up", "up"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 0, y: 2, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 10, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 0, y: 0, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 5, y: 0, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 10, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 8, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 3, content: 'nothing', color: 'text-blue-500' },
    { x: 0, y: 7, content: 'redAmulet', color: 'text-red-500' },
    { x: 10, y: 7, content: 'nothing', color: 'text-blue-500' },
    { x: 0, y: 11, content: 'nothing', color: 'text-blue-500' },
    { x: 8, y: 11, content: 'blueAmulet', color: 'text-blue-500' },
  ],
  blocks: [{x:1,y:0}, {x:2,y:0}, {x:3,y:0}, {x:4,y:0}, {x:6,y:0}, {x:7,y:0}, {x:9,y:0}, {x:10,y:0}, {x:1,y:1}, {x:2,y:1}, {x:3,y:1}, {x:4,y:1}, {x:6,y:1}, {x:7,y:1}, {x:9,y:1}, {x:10,y:1}, {x:1,y:2}, {x:2,y:2}, {x:3,y:2}, {x:4,y:2}, {x:6,y:2}, {x:7,y:2}, {x:9,y:2}, {x:10,y:2}, {x:0,y:4}, {x:1,y:4}, {x:2,y:4}, {x:3,y:4}, {x:4,y:4}, {x:6,y:4}, {x:7,y:4}, {x:9,y:4}, {x:10,y:4}, {x:0,y:5}, {x:1,y:5}, {x:2,y:5}, {x:3,y:5}, {x:4,y:5}, {x:6,y:5}, {x:7,y:5}, {x:9,y:5}, {x:10,y:5}, {x:0,y:6}, {x:1,y:6}, {x:2,y:6}, {x:3,y:6}, {x:4,y:6}, {x:6,y:6}, {x:7,y:6}, {x:9,y:6}, {x:10,y:6}, {x:6,y:7}, {x:7,y:7}, {x:0,y:8}, {x:1,y:8}, {x:2,y:8}, {x:3,y:8}, {x:4,y:8}, {x:6,y:8}, {x:7,y:8}, {x:9,y:8}, {x:10,y:8}, {x:1,y:10}, {x:2,y:10}, {x:3,y:10}, {x:4,y:10}, {x:5,y:10}, {x:6,y:10}, {x:7,y:10}, {x:9,y:10}, {x:1,y:11}, {x:2,y:11}, {x:3,y:11}, {x:4,y:11}, {x:5,y:11}, {x:6,y:11}, {x:7,y:11}, {x:9,y:11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 65,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
},
level442: {
  player: { x: 5, y: 7 },
  npc: { 
    x: 5, 
    y: 3, 
    movements: {
      experienced1: { path: ["right", "right", "right", "down", "down", "down", "down", "down", "down", "down", "up", "right", "right", "down", "down"], goal: 1 },
      experienced2: { path: ["up", "up", "up"], goal: 2 },
      novice1: { path: ["up"], goal: 1 },
      novice2: { path: ["up", "up", "up", "up"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 0, y: 2, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 10, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 0, y: 0, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 5, y: 0, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 10, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 8, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 3, content: 'nothing', color: 'text-blue-500' },
    { x: 0, y: 7, content: 'redAmulet', color: 'text-red-500' },
    { x: 10, y: 7, content: 'nothing', color: 'text-blue-500' },
    { x: 0, y: 11, content: 'nothing', color: 'text-blue-500' },
    { x: 8, y: 11, content: 'blueAmulet', color: 'text-blue-500' },
  ],
  blocks: [{x:1,y:0}, {x:2,y:0}, {x:3,y:0}, {x:4,y:0}, {x:6,y:0}, {x:7,y:0}, {x:9,y:0}, {x:10,y:0}, {x:1,y:1}, {x:2,y:1}, {x:3,y:1}, {x:4,y:1}, {x:6,y:1}, {x:7,y:1}, {x:9,y:1}, {x:10,y:1}, {x:1,y:2}, {x:2,y:2}, {x:3,y:2}, {x:4,y:2}, {x:6,y:2}, {x:7,y:2}, {x:9,y:2}, {x:10,y:2}, {x:0,y:4}, {x:1,y:4}, {x:2,y:4}, {x:3,y:4}, {x:4,y:4}, {x:6,y:4}, {x:7,y:4}, {x:9,y:4}, {x:10,y:4}, {x:0,y:5}, {x:1,y:5}, {x:2,y:5}, {x:3,y:5}, {x:4,y:5}, {x:6,y:5}, {x:7,y:5}, {x:9,y:5}, {x:10,y:5}, {x:0,y:6}, {x:1,y:6}, {x:2,y:6}, {x:3,y:6}, {x:4,y:6}, {x:6,y:6}, {x:7,y:6}, {x:9,y:6}, {x:10,y:6}, {x:6,y:7}, {x:7,y:7}, {x:0,y:8}, {x:1,y:8}, {x:2,y:8}, {x:3,y:8}, {x:4,y:8}, {x:6,y:8}, {x:7,y:8}, {x:9,y:8}, {x:10,y:8}, {x:1,y:10}, {x:2,y:10}, {x:3,y:10}, {x:4,y:10}, {x:5,y:10}, {x:6,y:10}, {x:7,y:10}, {x:9,y:10}, {x:1,y:11}, {x:2,y:11}, {x:3,y:11}, {x:4,y:11}, {x:5,y:11}, {x:6,y:11}, {x:7,y:11}, {x:9,y:11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 60,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
},
level511: {
  player: { x: 2, y: 7 },
  npc: { 
    x: 2, 
    y: 5, 
    movements: {
      experienced1: { path: ["up", "up", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "down", "down", "left", "left", "left", "left", "left", "left", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "down", "down"], goal: 1 },
      experienced2: { path: ["left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"], goal: 2 },
      novice1: { path: ["up"], goal: 1 },
      novice2: { path: ["left", "right", "down", "down", "down", "down", "right", "right", "right", "right", "right", "right", "right", "right"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 9, y: 9, requiredItems: ["redAmulet"], color: 'text-red-500' },
    { x: 0, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 4, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 10, y: 9, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 0, y: 11, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 4, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 0, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 3, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 6, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 0, y: 5, content: 'redAmulet', color: 'text-red-500' },
    { x: 10, y: 6, content: 'nothing', color: 'text-blue-500' },
  ],
  blocks: [{x:1,y:0}, {x:2,y:0}, {x:4,y:0}, {x:5,y:0}, {x:7,y:0}, {x:8,y:0}, {x:9,y:0}, {x:1,y:1}, {x:2,y:1}, {x:4,y:1}, {x:5,y:1}, {x:7,y:1}, {x:8,y:1}, {x:9,y:1}, {x:1,y:2}, {x:2,y:2}, {x:4,y:2}, {x:5,y:2}, {x:7,y:2}, {x:8,y:2}, {x:9,y:2}, {x:0,y:4}, {x:1,y:4}, {x:3,y:4}, {x:4,y:4}, {x:5,y:4}, {x:6,y:4}, {x:7,y:4}, {x:8,y:4}, {x:9,y:4}, {x:3,y:5}, {x:4,y:5}, {x:5,y:5}, {x:6,y:5}, {x:7,y:5}, {x:8,y:5}, {x:9,y:5}, {x:0,y:6}, {x:1,y:6}, {x:3,y:6}, {x:4,y:6}, {x:5,y:6}, {x:6,y:6}, {x:7,y:6}, {x:8,y:6}, {x:9,y:6}, {x:0,y:7}, {x:1,y:7}, {x:3,y:7}, {x:4,y:7}, {x:5,y:7}, {x:6,y:7}, {x:7,y:7}, {x:8,y:7}, {x:9,y:7}, {x:10,y:7}, {x:0,y:8}, {x:1,y:8}, {x:3,y:8}, {x:4,y:8}, {x:5,y:8}, {x:6,y:8}, {x:7,y:8}, {x:8,y:8}, {x:9,y:8}, {x:10,y:8}, {x:1,y:10}, {x:2,y:10}, {x:3,y:10}, {x:5,y:10}, {x:6,y:10}, {x:7,y:10}, {x:8,y:10}, {x:9,y:10}, {x:10,y:10}, {x:1,y:11}, {x:2,y:11}, {x:3,y:11}, {x:5,y:11}, {x:6,y:11}, {x:7,y:11}, {x:8,y:11}, {x:9,y:11}, {x:10,y:11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 80,
  goal: {
    type: 'C',
    description: 'Find and obtain Treasure C'
  }
},
level521: {
  player: { x: 0, y: 9 },
  npc: { 
    x: 4, 
    y: 9, 
    movements: {
      experienced1: { path: ["up", "up", "up", "up", "up", "up", "left", "left", "left", "left", "up", "up", "down", "down", "right", "right", "right", "right", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "up", "up"], goal: 1 },
      experienced2: { path: ['down', 'down', 'right', 'right', 'right', 'right', 'right', 'right'], goal: 2 },
      novice1: { path: ["up"], goal: 1 },
      novice2: { path: ['down', 'down', 'right', 'right', 'right', 'right', 'right', 'right'], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 0, y: 8, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 0, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 0, y: 7, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 0, y: 11, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 10, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 0, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 10, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 5, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 7, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 9, content: 'nothing', color: 'text-blue-500' },
  ],
  blocks: [{ x: 1, y: 0}, { x: 2, y: 0}, { x: 3, y: 0}, { x: 4, y: 0}, { x: 5, y: 0}, { x: 6, y: 0}, { x: 7, y: 0}, { x: 8, y: 0}, { x: 9, y: 0}, { x: 1, y: 1}, { x: 2, y: 1}, { x: 3, y: 1}, { x: 4, y: 1}, { x: 5, y: 1}, { x: 6, y: 1}, { x: 7, y: 1}, { x: 8, y: 1}, { x: 9, y: 1}, { x: 1, y: 2}, { x: 2, y: 2}, { x: 3, y: 2}, { x: 4, y: 2}, { x: 5, y: 2}, { x: 6, y: 2}, { x: 7, y: 2}, { x: 8, y: 2}, { x: 9, y: 2}, { x: 0, y: 4}, { x: 1, y: 4}, { x: 2, y: 4}, { x: 3, y: 4}, { x: 5, y: 4}, { x: 6, y: 4}, { x: 7, y: 4}, { x: 8, y: 4}, { x: 9, y: 4}, { x: 10, y: 4}, { x: 0, y: 5}, { x: 1, y: 5}, { x: 2, y: 5}, { x: 3, y: 5}, { x: 0, y: 6}, { x: 1, y: 6}, { x: 2, y: 6}, { x: 3, y: 6}, { x: 5, y: 6}, { x: 6, y: 6}, { x: 7, y: 6}, { x: 8, y: 6}, { x: 9, y: 6}, { x: 10, y: 6}, { x: 1, y: 7}, { x: 2, y: 7}, { x: 3, y: 7}, { x: 1, y: 8}, { x: 2, y: 8}, { x: 3, y: 8}, { x: 5, y: 8}, { x: 6, y: 8}, { x: 7, y: 8}, { x: 8, y: 8}, { x: 9, y: 8}, { x: 10, y: 8}, { x: 1, y: 10}, { x: 2, y: 10}, { x: 3, y: 10}, { x: 5, y: 10}, { x: 6, y: 10}, { x: 7, y: 10}, { x: 8, y: 10}, { x: 9, y: 10}, { x: 10, y: 10}, { x: 1, y: 11}, { x: 2, y: 11}, { x: 3, y: 11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 85,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
},
level531: {
  player: { x: 5, y: 7 },
  npc: { 
    x: 5, 
    y: 5, 
    movements: {
      experienced1: { path: ["up", "up", "right", "right", "up", "up", "up", "right", "right", "left", "left", "down", "down", "down", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right"], goal: 1 },
      // experienced2: { path: ["up", "up", "right", "right", "up", "up", "up", "right", "right", "left", "left", "down", "down", "down", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "left", "down", "up", "right", "right", "right", "right", "right", "up", "up", "up", "up", "left", "left", "left", "left"], goal: 2 },
      experienced2: { path: ["left", "left", "left", "left", "left"], goal: 2 },
      novice1: { path: ["up"], goal: 1 },
      novice2: { path: ["up", "up", "right", "right", "up", "up", "up", "right", "right", "left", "left", "down", "down", "down", "left", "left", "down", "down", "down", "down", "down", "down", "left", "left", "left", "left", "left", "down", "up", "right", "right", "right", "right", "right", "up", "up", "up", "up", "left", "left", "left", "left"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 9, y: 9, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 5, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 0, y: 5, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 10, y: 9, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 5, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 0, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 4, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 10, y: 3, content: 'nothing', color: 'text-blue-500' },
    { x: 0, y: 11, content: 'redAmulet', color: 'text-red-500' },
  ],
  blocks: [{ x: 1, y: 0}, { x: 2, y: 0}, { x: 3, y: 0}, { x: 1, y: 1}, { x: 2, y: 1}, { x: 3, y: 1}, { x: 4, y: 1}, { x: 5, y: 1}, { x: 6, y: 1}, { x: 8, y: 1}, { x: 9, y: 1}, { x: 10, y: 1}, { x: 1, y: 2}, { x: 2, y: 2}, { x: 3, y: 2}, { x: 4, y: 2}, { x: 5, y: 2}, { x: 6, y: 2}, { x: 8, y: 2}, { x: 9, y: 2}, { x: 10, y: 2}, { x: 0, y: 4}, { x: 1, y: 4}, { x: 2, y: 4}, { x: 3, y: 4}, { x: 4, y: 4}, { x: 6, y: 4}, { x: 7, y: 4}, { x: 8, y: 4}, { x: 9, y: 4}, { x: 10, y: 4}, { x: 0, y: 6}, { x: 1, y: 6}, { x: 2, y: 6}, { x: 3, y: 6}, { x: 4, y: 6}, { x: 6, y: 6}, { x: 7, y: 6}, { x: 8, y: 6}, { x: 9, y: 6}, { x: 10, y: 6}, { x: 0, y: 7}, { x: 1, y: 7}, { x: 2, y: 7}, { x: 3, y: 7}, { x: 4, y: 7}, { x: 6, y: 7}, { x: 7, y: 7}, { x: 8, y: 7}, { x: 9, y: 7}, { x: 10, y: 7}, { x: 0, y: 8}, { x: 1, y: 8}, { x: 2, y: 8}, { x: 3, y: 8}, { x: 4, y: 8}, { x: 6, y: 8}, { x: 7, y: 8}, { x: 8, y: 8}, { x: 9, y: 8}, { x: 10, y: 8}, { x: 1, y: 10}, { x: 2, y: 10}, { x: 3, y: 10}, { x: 4, y: 10}, { x: 6, y: 10}, { x: 7, y: 10}, { x: 8, y: 10}, { x: 9, y: 10}, { x: 10, y: 10}, { x: 1, y: 11}, { x: 2, y: 11}, { x: 3, y: 11}, { x: 4, y: 11}, { x: 6, y: 11}, { x: 7, y: 11}, { x: 8, y: 11}, { x: 9, y: 11}, { x: 10, y: 11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 70,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
},
level532: {
  player: { x: 5, y: 7 },
  npc: { 
    x: 7, 
    y: 3, 
    movements: {
      experienced1: { path: ["up", "up", "up", "right", "right", "left", "left", "down", "down", "down", "left", "left", "down", "down", "down", "down", "down", "down", "right", "right", "right", "right", "right"], goal: 1 },
      // experienced2: { path: ["up", "up", "up", "right", "right", "left", "left", "down", "down", "down", "left", "left", "down", "down", "down", "down", "down", "down", "down", "down"], goal: 2 },
      experienced2: { path: ["left", "left", "down", "down", "left", "left", "left", "left", "left"], goal: 2 },
      novice1: { path: ["up"], goal: 1 },
      novice2: { path: ["up", "up", "up", "right", "right", "left", "left", "down", "down", "down", "left", "left", "down", "down", "down", "down", "down", "down", "down", "down"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 9, y: 9, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 5, y: 10, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 0, y: 5, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 10, y: 9, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 5, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 0, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 4, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 10, y: 3, content: 'nothing', color: 'text-blue-500' },
    { x: 0, y: 11, content: 'redAmulet', color: 'text-red-500' },
  ],
  blocks: [{ x: 1, y: 0}, { x: 2, y: 0}, { x: 3, y: 0}, { x: 1, y: 1}, { x: 2, y: 1}, { x: 3, y: 1}, { x: 4, y: 1}, { x: 5, y: 1}, { x: 6, y: 1}, { x: 8, y: 1}, { x: 9, y: 1}, { x: 10, y: 1}, { x: 1, y: 2}, { x: 2, y: 2}, { x: 3, y: 2}, { x: 4, y: 2}, { x: 5, y: 2}, { x: 6, y: 2}, { x: 8, y: 2}, { x: 9, y: 2}, { x: 10, y: 2}, { x: 0, y: 4}, { x: 1, y: 4}, { x: 2, y: 4}, { x: 3, y: 4}, { x: 4, y: 4}, { x: 6, y: 4}, { x: 7, y: 4}, { x: 8, y: 4}, { x: 9, y: 4}, { x: 10, y: 4}, { x: 0, y: 6}, { x: 1, y: 6}, { x: 2, y: 6}, { x: 3, y: 6}, { x: 4, y: 6}, { x: 6, y: 6}, { x: 7, y: 6}, { x: 8, y: 6}, { x: 9, y: 6}, { x: 10, y: 6}, { x: 0, y: 7}, { x: 1, y: 7}, { x: 2, y: 7}, { x: 3, y: 7}, { x: 4, y: 7}, { x: 6, y: 7}, { x: 7, y: 7}, { x: 8, y: 7}, { x: 9, y: 7}, { x: 10, y: 7}, { x: 0, y: 8}, { x: 1, y: 8}, { x: 2, y: 8}, { x: 3, y: 8}, { x: 4, y: 8}, { x: 6, y: 8}, { x: 7, y: 8}, { x: 8, y: 8}, { x: 9, y: 8}, { x: 10, y: 8}, { x: 1, y: 10}, { x: 2, y: 10}, { x: 3, y: 10}, { x: 4, y: 10}, { x: 6, y: 10}, { x: 7, y: 10}, { x: 8, y: 10}, { x: 9, y: 10}, { x: 10, y: 10}, { x: 1, y: 11}, { x: 2, y: 11}, { x: 3, y: 11}, { x: 4, y: 11}, { x: 6, y: 11}, { x: 7, y: 11}, { x: 8, y: 11}, { x: 9, y: 11}, { x: 10, y: 11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 65,
  goal: {
    type: 'B',
    description: 'Find and obtain Treasure B'
  }
},
level541: {
  player: { x: 0, y: 9 },
  npc: { 
    x: 5, 
    y: 5, 
    movements: {
      experienced1: { path: ["up", "up", "up", "left", "left", "up", "down", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"], goal: 1 },
      experienced2: { path: ["right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"], goal: 2 },
      novice1: { path: ["up", "up", "up", "right", "right", "up", "down", "left", "left", "left", "left", "up", "down", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"], goal: 1 },
      novice2: { path: ["right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 10, y: 10, requiredItems: ["redAmulet"], color: 'text-red-500' },
    { x: 2, y: 5, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 0, y: 5, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 5, y: 11, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 10, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 3, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 4, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 7, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 8, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 5, content: 'redAmulet', color: 'text-red-500' },
  ],
  blocks: [{x:0,y:0}, {x:1,y:0}, {x:2,y:0}, {x:5,y:0}, {x:6,y:0}, {x:9,y:0}, {x:10,y:0}, {x:0,y:1}, {x:1,y:1}, {x:2,y:1}, {x:5,y:1}, {x:6,y:1}, {x:9,y:1}, {x:10,y:1}, {x:0,y:3}, {x:1,y:3}, {x:2,y:3}, {x:3,y:3}, {x:4,y:3}, {x:8,y:3}, {x:9,y:3}, {x:10,y:3}, {x:0,y:4}, {x:1,y:4}, {x:2,y:4}, {x:3,y:4}, {x:4,y:4}, {x:6,y:4}, {x:7,y:4}, {x:8,y:4}, {x:9,y:4}, {x:10,y:4}, {x:0,y:6}, {x:1,y:6}, {x:2,y:6}, {x:3,y:6}, {x:4,y:6}, {x:6,y:6}, {x:7,y:6}, {x:8,y:6}, {x:9,y:6}, {x:10,y:6}, {x:0,y:7}, {x:1,y:7}, {x:2,y:7}, {x:3,y:7}, {x:4,y:7}, {x:6,y:7}, {x:7,y:7}, {x:8,y:7}, {x:9,y:7}, {x:10,y:7}, {x:0,y:8}, {x:1,y:8}, {x:2,y:8}, {x:3,y:8}, {x:4,y:8}, {x:6,y:8}, {x:7,y:8}, {x:8,y:8}, {x:9,y:8}, {x:10,y:8}, {x:0,y:10}, {x:1,y:10}, {x:2,y:10}, {x:3,y:10}, {x:4,y:10}, {x:6,y:10}, {x:7,y:10}, {x:8,y:10}, {x:9,y:10}, {x:0,y:11}, {x:1,y:11}, {x:2,y:11}, {x:3,y:11}, {x:4,y:11}, {x:6,y:11}, {x:7,y:11}, {x:8,y:11}, {x:9,y:11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 70,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
},
level542: {
  player: { x: 0, y: 9 },
  npc: { 
    x: 5, 
    y: 2, 
    movements: {
      experienced1: { path: ["left", "left", "up", "down", "right", "right", "down", "down", "down", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left"], goal: 1 },
      experienced2: { path: ["down", "down", "down", "down", "down", "down", "down", "down", "down"], goal: 2 },
      novice1: { path: ["right", "right", "up", "down", "left", "left", "left", "left", "up", "down", "right", "right", "down", "down", "down", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "left"], goal: 1 },
      novice2: { path: ["down", "down", "down", "down", "down", "down", "down", "down", "down", "down"], goal: 2 },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 1, y: 5, requiredItems: ["redAmulet"], color: 'text-red-500' },
    { x: 2, y: 5, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 0, y: 5, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 5, y: 11, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 10, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 3, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 4, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 7, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 8, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 5, content: 'redAmulet', color: 'text-red-500' },
  ],
  blocks: [{x:0,y:0}, {x:1,y:0}, {x:2,y:0}, {x:5,y:0}, {x:6,y:0}, {x:9,y:0}, {x:10,y:0}, {x:0,y:1}, {x:1,y:1}, {x:2,y:1}, {x:5,y:1}, {x:6,y:1}, {x:9,y:1}, {x:10,y:1}, {x:0,y:3}, {x:1,y:3}, {x:2,y:3}, {x:3,y:3}, {x:4,y:3}, {x:8,y:3}, {x:9,y:3}, {x:10,y:3}, {x:0,y:4}, {x:1,y:4}, {x:2,y:4}, {x:3,y:4}, {x:4,y:4}, {x:6,y:4}, {x:7,y:4}, {x:8,y:4}, {x:9,y:4}, {x:10,y:4}, {x:0,y:6}, {x:1,y:6}, {x:2,y:6}, {x:3,y:6}, {x:4,y:6}, {x:6,y:6}, {x:7,y:6}, {x:8,y:6}, {x:9,y:6}, {x:10,y:6}, {x:0,y:7}, {x:1,y:7}, {x:2,y:7}, {x:3,y:7}, {x:4,y:7}, {x:6,y:7}, {x:7,y:7}, {x:8,y:7}, {x:9,y:7}, {x:10,y:7}, {x:0,y:8}, {x:1,y:8}, {x:2,y:8}, {x:3,y:8}, {x:4,y:8}, {x:6,y:8}, {x:7,y:8}, {x:8,y:8}, {x:9,y:8}, {x:10,y:8}, {x:0,y:10}, {x:1,y:10}, {x:2,y:10}, {x:3,y:10}, {x:4,y:10}, {x:6,y:10}, {x:7,y:10}, {x:8,y:10}, {x:9,y:10}, {x:0,y:11}, {x:1,y:11}, {x:2,y:11}, {x:3,y:11}, {x:4,y:11}, {x:6,y:11}, {x:7,y:11}, {x:8,y:11}, {x:9,y:11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 80,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
},
level543: {
  player: { x: 3, y: 2 },
  npc: { 
    x: 0, 
    y: 9, 
    movements: {
      experienced1: {
        path: ["right", "right", "right", "right", "right", "up", "up", "up", "up", "up", "up", "up", "left", "left", "up", "down", "right", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
        goal: 0
      },
      experienced2: {
        path: ["right", "right", "right", "right", "right", "up", "up", "up", "up", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
        goal: 0
      },
      novice1: {
        path: [],
        goal: 0
      },
      novice2: {
        path: [],
        goal: 0
      },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 2, y: 5, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 10, y: 10, requiredItems: ["redAmulet"], color: 'text-red-500' },
  ],
  treasurePots: [
    { x: 0, y: 5, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 5, y: 11, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 10, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 3, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 0, y: 2, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 5, content: 'redAmulet', color: 'text-red-500' },
  ],
  blocks: [{ x: 0, y: 0}, { x: 1, y: 0}, { x: 2, y: 0}, { x: 4, y: 0}, { x: 5, y: 0}, { x: 6, y: 0}, { x: 7, y: 0}, { x: 8, y: 0}, { x: 9, y: 0}, { x: 10, y: 0}, { x: 0, y: 1}, { x: 1, y: 1}, { x: 2, y: 1}, { x: 4, y: 1}, { x: 5, y: 1}, { x: 6, y: 1}, { x: 7, y: 1}, { x: 8, y: 1}, { x: 9, y: 1}, { x: 10, y: 1}, { x: 0, y: 3}, { x: 1, y: 3}, { x: 2, y: 3}, { x: 3, y: 3}, { x: 4, y: 3}, { x: 6, y: 3}, { x: 7, y: 3}, { x: 8, y: 3}, { x: 9, y: 3}, { x: 10, y: 3}, { x: 0, y: 4}, { x: 1, y: 4}, { x: 2, y: 4}, { x: 3, y: 4}, { x: 4, y: 4}, { x: 6, y: 4}, { x: 7, y: 4}, { x: 8, y: 4}, { x: 9, y: 4}, { x: 10, y: 4}, { x: 0, y: 6}, { x: 1, y: 6}, { x: 2, y: 6}, { x: 3, y: 6}, { x: 4, y: 6}, { x: 6, y: 6}, { x: 7, y: 6}, { x: 8, y: 6}, { x: 9, y: 6}, { x: 10, y: 6}, { x: 0, y: 7}, { x: 1, y: 7}, { x: 2, y: 7}, { x: 3, y: 7}, { x: 4, y: 7}, { x: 6, y: 7}, { x: 7, y: 7}, { x: 8, y: 7}, { x: 9, y: 7}, { x: 10, y: 7}, { x: 0, y: 8}, { x: 1, y: 8}, { x: 2, y: 8}, { x: 3, y: 8}, { x: 4, y: 8}, { x: 6, y: 8}, { x: 7, y: 8}, { x: 8, y: 8}, { x: 9, y: 8}, { x: 10, y: 8}, { x: 0, y: 10}, { x: 1, y: 10}, { x: 2, y: 10}, { x: 3, y: 10}, { x: 4, y: 10}, { x: 6, y: 10}, { x: 7, y: 10}, { x: 8, y: 10}, { x: 9, y: 10}, { x: 0, y: 11}, { x: 1, y: 11}, { x: 2, y: 11}, { x: 3, y: 11}, { x: 4, y: 11}, { x: 6, y: 11}, { x: 7, y: 11}, { x: 8, y: 11}, { x: 9, y: 11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 50,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
},
level371: {
  player: { x: 0, y: 3 },
  npc: { 
    x: 10, 
    y: 8, 
    movements: {
      experienced1: {
        path: ["left", "left", "left", "left", "left", "left", "left", "up", "up", "up", "up", "up", "left", "left", "left", "down", "up", "right", "right", "right", "right", "right", "right", "right", "right", "right", "up", "up", "up"],
        goal: 0
      },
      experienced2: {
        path: ["left", "left", "left", "left", "left", "down", "down", "down"],
        goal: 0
      },
      novice1: {
        path: [],
        goal: 0
      },
      novice2: {
        path: [],
        goal: 0
      },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 9, y: 1, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
  ],
  treasurePots: [
    { x: 9, y: 0, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 5, y: 11, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 9, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 0, y: 1, content: 'nothing', color: 'text-blue-500' },
    { x: 0, y: 5, content: 'blueAmulet', color: 'text-blue-500' },
  ],
  blocks: [{ x: 0, y: 0}, { x: 1, y: 0}, { x: 2, y: 0}, { x: 3, y: 0}, { x: 4, y: 0}, { x: 5, y: 0}, { x: 6, y: 0}, { x: 7, y: 0}, { x: 8, y: 0}, { x: 10, y: 0}, { x: 1, y: 1}, { x: 2, y: 1}, { x: 3, y: 1}, { x: 4, y: 1}, { x: 5, y: 1}, { x: 6, y: 1}, { x: 7, y: 1}, { x: 8, y: 1}, { x: 10, y: 1}, { x: 1, y: 2}, { x: 2, y: 2}, { x: 3, y: 2}, { x: 4, y: 2}, { x: 5, y: 2}, { x: 6, y: 2}, { x: 7, y: 2}, { x: 8, y: 2}, { x: 10, y: 2}, { x: 10, y: 3}, { x: 1, y: 4}, { x: 2, y: 4}, { x: 4, y: 4}, { x: 5, y: 4}, { x: 6, y: 4}, { x: 7, y: 4}, { x: 8, y: 4}, { x: 9, y: 4}, { x: 10, y: 4}, { x: 1, y: 5}, { x: 2, y: 5}, { x: 4, y: 5}, { x: 5, y: 5}, { x: 6, y: 5}, { x: 7, y: 5}, { x: 8, y: 5}, { x: 9, y: 5}, { x: 10, y: 5}, { x: 0, y: 6}, { x: 1, y: 6}, { x: 2, y: 6}, { x: 4, y: 6}, { x: 5, y: 6}, { x: 6, y: 6}, { x: 7, y: 6}, { x: 8, y: 6}, { x: 9, y: 6}, { x: 10, y: 6}, { x: 0, y: 7}, { x: 1, y: 7}, { x: 2, y: 7}, { x: 4, y: 7}, { x: 5, y: 7}, { x: 6, y: 7}, { x: 7, y: 7}, { x: 8, y: 7}, { x: 9, y: 7}, { x: 10, y: 7}, { x: 0, y: 9}, { x: 1, y: 9}, { x: 2, y: 9}, { x: 3, y: 9}, { x: 4, y: 9}, { x: 6, y: 9}, { x: 7, y: 9}, { x: 8, y: 9}, { x: 10, y: 9}, { x: 0, y: 10}, { x: 1, y: 10}, { x: 2, y: 10}, { x: 3, y: 10}, { x: 4, y: 10}, { x: 6, y: 10}, { x: 7, y: 10}, { x: 8, y: 10}, { x: 10, y: 10}, { x: 0, y: 11}, { x: 1, y: 11}, { x: 2, y: 11}, { x: 3, y: 11}, { x: 4, y: 11}, { x: 6, y: 11}, { x: 7, y: 11}, { x: 8, y: 11}, { x: 10, y: 11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 45,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
},
level544: {
  player: { x: 5, y: 1 },
  npc: { 
    x: 0, 
    y: 9, 
    movements: {
      experienced1: {
        path: ["right", "right", "right", "right", "right", "up", "up", "up", "up", "up", "up", "up", "left", "up", "down", "right", "down", "down", "down", "left", "left", "left", "left", "left"],
        goal: 0
      },
      experienced2: {
        path: ["right", "right", "right", "right", "right", "up", "up", "up", "up", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
        goal: 0
      },
      novice1: {
        path: [],
        goal: 0
      },
      novice2: {
        path: ["right", "right", "right", "right", "right", "up", "up", "up", "up", "right", "right", "right", "right", "left", "left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right", "right", "down", "down"],
        goal: 0
      },
    },
    currentMovementIndex: 0 
  },
  barriers: [
    { x: 2, y: 5, requiredItems: ["blueAmulet"], color: 'text-blue-500' },
    { x: 10, y: 10, requiredItems: ["redAmulet"], color: 'text-red-500' },
  ],
  treasurePots: [
    { x: 0, y: 5, type: 'A', color: 'text-yellow-500', label: 'A' },
    { x: 5, y: 11, type: 'B', color: 'text-yellow-500', label: 'B' },
    { x: 10, y: 11, type: 'C', color: 'text-yellow-500', label: 'C' },
  ],
  wizards: [
    { x: 4, y: 0, content: 'blueAmulet', color: 'text-blue-500' },
    { x: 6, y: 0, content: 'nothing', color: 'text-blue-500' },
    { x: 10, y: 5, content: 'redAmulet', color: 'text-red-500' },
  ],
  blocks: [{ x: 0, y: 0}, { x: 1, y: 0}, { x: 2, y: 0}, { x: 3, y: 0}, { x: 5, y: 0}, { x: 7, y: 0}, { x: 8, y: 0}, { x: 9, y: 0}, { x: 10, y: 0}, { x: 0, y: 1}, { x: 1, y: 1}, { x: 2, y: 1}, { x: 3, y: 1}, { x: 7, y: 1}, { x: 8, y: 1}, { x: 9, y: 1}, { x: 10, y: 1}, { x: 0, y: 3}, { x: 1, y: 3}, { x: 2, y: 3}, { x: 3, y: 3}, { x: 4, y: 3}, { x: 6, y: 3}, { x: 7, y: 3}, { x: 8, y: 3}, { x: 9, y: 3}, { x: 10, y: 3}, { x: 0, y: 4}, { x: 1, y: 4}, { x: 2, y: 4}, { x: 3, y: 4}, { x: 4, y: 4}, { x: 6, y: 4}, { x: 7, y: 4}, { x: 8, y: 4}, { x: 9, y: 4}, { x: 10, y: 4}, { x: 0, y: 6}, { x: 1, y: 6}, { x: 2, y: 6}, { x: 3, y: 6}, { x: 4, y: 6}, { x: 6, y: 6}, { x: 7, y: 6}, { x: 8, y: 6}, { x: 9, y: 6}, { x: 10, y: 6}, { x: 0, y: 7}, { x: 1, y: 7}, { x: 2, y: 7}, { x: 3, y: 7}, { x: 4, y: 7}, { x: 6, y: 7}, { x: 7, y: 7}, { x: 8, y: 7}, { x: 9, y: 7}, { x: 10, y: 7}, { x: 0, y: 8}, { x: 1, y: 8}, { x: 2, y: 8}, { x: 3, y: 8}, { x: 4, y: 8}, { x: 6, y: 8}, { x: 7, y: 8}, { x: 8, y: 8}, { x: 9, y: 8}, { x: 10, y: 8}, { x: 0, y: 10}, { x: 1, y: 10}, { x: 2, y: 10}, { x: 3, y: 10}, { x: 4, y: 10}, { x: 6, y: 10}, { x: 7, y: 10}, { x: 8, y: 10}, { x: 9, y: 10}, { x: 0, y: 11}, { x: 1, y: 11}, { x: 2, y: 11}, { x: 3, y: 11}, { x: 4, y: 11}, { x: 6, y: 11}, { x: 7, y: 11}, { x: 8, y: 11}, { x: 9, y: 11}],
  inventory: [],
  openedBarriers: [],
  stepLimit: 40,
  goal: {
    type: 'A',
    description: 'Find and obtain Treasure A'
  }
},
};

export default mapData;
