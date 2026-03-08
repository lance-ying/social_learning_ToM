import { ref, push } from 'firebase/database';
import { db } from '../config/firebaseConfig';
import { GameState, Agent } from '@/types/game';
import { firebaseLogger } from './FirebaseLogger';
import { EXPERIMENT_TYPE } from '@/app/components/game-flow/introductions/config';

interface GameLog {
  type: string;
  data: any;
  timestamp?: any;
}

interface Wizard {
  x: number;
  y: number;
  content: string;
  interactedBy: number[]; // Array of agent IDs who have interacted with this wizard
}

const GAME_MESSAGES = {
  BLUE_AMULET: (agentId: number | null, agentColor?: string, experimentType?: string) => {
    // Generic message for exp4 when observing agents
    if (experimentType === 'exp4' && agentId !== null) {
      return `<img src="${agentColor === 'blue' ? '/icons/al.png' : '/icons/green_a.png'}" alt="Other Player" style="display: inline; width: 32px; height: 40px; vertical-align: middle; margin-right: 6px;"> interacted with a wizard!|${Date.now()}`;
    }
    // Original detailed message for other experiments and player's own interactions
    return `${agentId ? `<img src="${agentColor === 'blue' ? '/icons/al.png' : '/icons/green_a.png'}" alt="Other Player" style="display: inline; width: 32px; height: 40px; vertical-align: middle; margin-right: 6px;">` : 'You'} received a <span style="color: blue; font-weight: bold">blue amulet</span> from the <span style="color: blue">blue wizard</span>!|${Date.now()}`;
  },
  RED_AMULET: (agentId: number | null, agentColor?: string, experimentType?: string) => {
    if (experimentType === 'exp4' && agentId !== null) {
      return `<img src="${agentColor === 'blue' ? '/icons/al.png' : '/icons/green_a.png'}" alt="Other Player" style="display: inline; width: 32px; height: 40px; vertical-align: middle; margin-right: 6px;"> interacted with a wizard!|${Date.now()}`;
    }
    return `${agentId ? `<img src="${agentColor === 'blue' ? '/icons/al.png' : '/icons/green_a.png'}" alt="Other Player" style="display: inline; width: 32px; height: 40px; vertical-align: middle; margin-right: 6px;">` : 'You'} received a <span style="color: red; font-weight: bold">red amulet</span> from the <span style="color: red">red wizard</span>!|${Date.now()}`;
  },
  EMPTY_WIZARD: (agentId: number | null, agentColor?: string, experimentType?: string) => {
    if (experimentType === 'exp4' && agentId !== null) {
      return `<img src="${agentColor === 'blue' ? '/icons/al.png' : '/icons/green_a.png'}" alt="Other Player" style="display: inline; width: 32px; height: 40px; vertical-align: middle; margin-right: 6px;"> interacted with a wizard!|${Date.now()}`;
    }
    return `${agentId ? `<img src="${agentColor === 'blue' ? '/icons/al.png' : '/icons/green_a.png'}" alt="Other Player" style="display: inline; width: 32px; height: 40px; vertical-align: middle; margin-right: 6px;">` : 'You'} received <span style="font-weight: bold">nothing</span> from the <span style="color: blue">blue wizard</span>!|${Date.now()}`;
  },
  ALREADY_INTERACTED: "You've already interacted with this wizard.",
  BARRIER_BLUE: (agent: string) => 
    `${agent} need a <span style="color: blue; font-weight: bold">blue amulet</span> to pass this barrier.`,
  BARRIER_RED: (agent: string) => 
    `${agent} need a <span style="color: red; font-weight: bold">red amulet</span> to pass this barrier.`,
  BARRIER_PASSED: (agent: string) => 
    `${agent} passed through the barrier!`,
  AGENT_PATH_COMPLETE: (agentId: number, agentColor?: string) => 
    `<span style="color: green"><img src="${agentColor === 'blue' ? '/icons/al.png' : '/icons/green_a.png'}" alt="Other Player" style="display: inline; width: 32px; height: 40px; vertical-align: middle; margin-right: 6px;"> has reached the end of their path!</span>|${Date.now()}`,
};

export class GameLogic {
  // Configurable trail length - change this value to adjust trail length for all agents
  private static readonly TRAIL_LENGTH = 20;

  static async addGameLog(gameSessionId: string, log: GameLog) {
    if (!gameSessionId) return;
    
    try {
      // For Realtime Database, we'll push to a different path structure
      const logsRef = ref(db, `gameSessions/${gameSessionId}/logs`);
      await push(logsRef, {
        ...log,
        timestamp: Date.now() // Using client timestamp for Realtime Database
      });
    } catch (error) {
      console.error("Failed to add game log:", error);
    }
  }

  static handleMove(
    direction: string,
    state: GameState,
    callbacks: {
      setMessage: (message: string) => void;
      setActionLog: (log: any[]) => void;
      setGameWon: (won: boolean) => void;
    },
    gameSessionId?: string,
    currentLevel?: string,
    agentMovesPerPlayerAction: number = 0, // Keep parameter for compatibility but ignore it
    isDevMode: boolean = false
  ): GameState {
    const { setMessage, setActionLog, setGameWon } = callbacks;

    // Log player action to Firebase
    if (firebaseLogger.isLoggingEnabled() && currentLevel) {
      firebaseLogger.logPlayerAction(
        direction,
        currentLevel,
        { x: state.player.x, y: state.player.y }
      ).catch(error => console.error('Failed to log player action:', error));
    }

    // Pass the callbacks to calculateNewState - agents will NOT move on player movement
    return this.calculateNewState(state, direction, callbacks, currentLevel, isDevMode);
  }

  static handleObserve(
    agentId: number,
    state: GameState,
    callbacks: {
      setMessage: (message: string) => void;
      setState: (state: GameState) => void;
    },
    gameSessionId?: string,
    npcMovesPerObservation: number = 1, // Changed from default to 1
    currentLevel?: string,
    isDevMode: boolean = false
  ) {
    const { setMessage, setState } = callbacks;

    const agent = state.agents.find((a: { id: number }) => a.id === agentId);
    if (!agent || agent.observesRemaining <= 0) {
      setMessage(`<span style="color: red">No more observations remaining for this agent.</span>`);
      return state;
    }

    // Check if agent has completed their path
    if (agent.currentPath && agent.currentMovementIndex >= agent.currentPath.length) {
      setMessage(`<span style="color: black">Player ${agentId} has completed their path. No more observations available.</span>`);
      return state;
    }

    const updatedAgent = this.moveNPCMultiple(
      agent,
      state.blocks,
      state.barriers,
      state.treasurePots,
      state.wizards,
      npcMovesPerObservation,
      callbacks,
      state
    );

    if (gameSessionId) {
      this.addGameLog(gameSessionId, {
        type: 'OBSERVE_ACTION',
        data: {
          agentId,
          newPosition: { x: updatedAgent.x, y: updatedAgent.y },
          movementIndex: updatedAgent.currentMovementIndex,
          pathType: agent.selectedPath,
          agentType: agent.type
        }
      });
    }

    // Log agent observe action to Firebase
    if (firebaseLogger.isLoggingEnabled() && currentLevel) {
      firebaseLogger.logAgentObserve(
        agentId,
        currentLevel,
        { x: updatedAgent.x, y: updatedAgent.y },
        agent.selectedPath || 'unknown',
        { x: state.player.x, y: state.player.y } // Include player position
      ).catch(error => console.error('Failed to log agent observe:', error));
    }

    const pointChange = isDevMode ? 1 : -1;
    const newState = {
      ...state,
      agents: state.agents.map(a =>
        a.id === agentId ? updatedAgent : a
      ),
      stepsRemaining: state.stepsRemaining + pointChange
    };

    setState(newState);
    return newState;
  }

  static initializeGame(state: GameState, preservePaths: boolean = false, isDevMode: boolean = false): GameState {
    const newState = JSON.parse(JSON.stringify(state));

    // Initialize wizards with empty interactedBy arrays
    newState.wizards = newState.wizards.map((wizard: Wizard) => ({
      ...wizard,
      interactedBy: []
    }));

    // Set resetTrails for player
    newState.player = {
      ...newState.player,
      resetTrails: true
    };

    newState.agents = newState.agents.map((agent: Agent) => {
      // Only randomize paths if preservePaths is false
      let pathType = agent.selectedPath;

      if (!preservePaths || !pathType) {
        // Get only the available path types for this agent
        const availablePathTypes = Object.keys(agent.movements);
        const randomIndex = Math.floor(Math.random() * availablePathTypes.length);
        pathType = availablePathTypes[randomIndex];
      }

      // Ensure the selected pathType exists in agent.movements
      if (!agent.movements[pathType as keyof typeof agent.movements]) {
        // Fallback to the first available path if the selected one doesn't exist
        const availablePathTypes = Object.keys(agent.movements);
        pathType = availablePathTypes[0];
      }

      return {
        ...agent,
        selectedPath: pathType,
        currentPath: agent.movements[pathType as keyof typeof agent.movements].path,
        currentMovementIndex: 0,
        type: agent.movements[pathType as keyof typeof agent.movements].type,
        resetTrails: true,  // Set to true when initializing
        isInteracting: false
      };
    });

    // Set resetTrails back to false after a short delay for both player and agents
    setTimeout(() => {
      newState.player.resetTrails = false;
      newState.agents = newState.agents.map((agent: Agent) => ({
        ...agent,
        resetTrails: false
      }));
    }, 100);

    // In dev mode, start with 0 points (counting up), otherwise use the level's configured value
    if (isDevMode) {
      newState.stepsRemaining = 0;
    }

    return newState;
  }

  static resetGame(state: GameState): GameState {
    // Reset the game state but preserve the current path selections
    return this.initializeGame(state, true);
  }

  private static handleWizardInteraction(
    state: GameState,
    wizardIndex: number,
    callbacks: any,
    gameSessionId?: string,
    currentLevel?: string,
    isDevMode: boolean = false
  ) {
    const { setMessage } = callbacks;
    const wizard = state.wizards[wizardIndex];

    if (wizard.interactedBy.includes(0)) { // 0 is the player's ID
      setMessage(GAME_MESSAGES.ALREADY_INTERACTED + `|${Date.now()}`);
      return state;
    }

    const newInventory = [...state.inventory];
    let interactionType = 'unknown';

    // Animation data for player interaction (agentId null means player)
    const animationData = JSON.stringify({
      wizardX: wizard.x,
      wizardY: wizard.y,
      agentId: null,
      wizardColor: wizard.color
    });

    if (wizard.content === 'blueAmulet') {
      newInventory.push(wizard.content);
      const fullMessage = `${GAME_MESSAGES.BLUE_AMULET(null)}|${animationData}`;
      console.log("GameLogic: Player wizard interaction:", fullMessage);
      setMessage(fullMessage);
      interactionType = 'blue_amulet_received';
    } else if (wizard.content === 'redAmulet') {
      newInventory.push(wizard.content);
      const fullMessage = `${GAME_MESSAGES.RED_AMULET(null)}|${animationData}`;
      console.log("GameLogic: Player wizard interaction:", fullMessage);
      setMessage(fullMessage);
      interactionType = 'red_amulet_received';
    } else if (wizard.content === 'nothing') {
      const fullMessage = `${GAME_MESSAGES.EMPTY_WIZARD(null)}|${animationData}`;
      console.log("GameLogic: Player wizard interaction:", fullMessage);
      setMessage(fullMessage);
      interactionType = 'nothing_received';
    }

    // Log wizard interaction to Firebase
    if (firebaseLogger.isLoggingEnabled() && currentLevel) {
      firebaseLogger.logInteraction(
        'wizard_interaction',
        {
          item: wizard.content !== 'nothing' ? wizard.content : null,
          wx: wizard.x,
          wy: wizard.y,
          px: state.player.x,
          py: state.player.y
        },
        currentLevel
      ).catch(error => console.error('Failed to log wizard interaction:', error));
    }

    const pointChange = isDevMode ? 5 : -5;
    return {
      ...state,
      inventory: newInventory,
      wizards: state.wizards.map((w, i) =>
        i === wizardIndex
          ? { ...w, interactedBy: [...w.interactedBy, 0] }
          : w
      ),
      stepsRemaining: state.stepsRemaining + pointChange
    };
  }

  private static handleTreasureInteraction(
    state: GameState,
    treasure: { x: number; y: number; type: string },
    newX: number,
    newY: number,
    callbacks: any,
    gameSessionId?: string,
    currentLevel?: string
  ): GameState {
    const { setMessage, setGameWon } = callbacks;
    const newInventory = [...state.inventory, treasure.type];
    
    // Check if the found treasure matches the goal
    if (state.goal.type === treasure.type) {
      setMessage(`<span style="color: green; font-weight: bold"> YOU WON! </span><span style="color: green">You found the correct treasure: ${treasure.type}!</span>`);
      setGameWon(true);
    } else {
      setMessage(`<span style="color: red; font-weight: bold">Wrong Treasure!</span><br/>You found treasure ${treasure.type}, but you're looking for treasure ${state.goal.type}!`);
    }
    
    if (gameSessionId) {
      this.addGameLog(gameSessionId, {
        type: 'TREASURE_INTERACTION',
        data: { treasureType: treasure.type, position: { x: newX, y: newY } }
      });
    }

    // Log treasure interaction to Firebase
    if (firebaseLogger.isLoggingEnabled() && currentLevel) {
      const isCorrect = state.goal.type === treasure.type;
      firebaseLogger.logInteraction(
        'treasure_interaction',
        {
          treasure: treasure.type,
          goal: state.goal.type,
          correct: isCorrect,
          outcome: isCorrect ? 'won' : 'wrong',
          x: newX,
          y: newY
        },
        currentLevel
      ).catch(error => console.error('Failed to log treasure interaction:', error));

      // Log level completion if won
      if (isCorrect && setGameWon) {
        firebaseLogger.logLevelComplete(
          currentLevel,
          true,
          100 - state.stepsRemaining
        ).catch(error => console.error('Failed to log level completion:', error));
      }
    }

    return {
      ...state,
      inventory: newInventory,
      player: { x: newX, y: newY, isPlayer: true, id: 0 },
      // Don't filter out the treasure anymore - keep it in the same position
    };
  }

  private static handleBarrierInteraction(
    state: GameState,
    barrier: { x: number; y: number; requiredItems: string[] },
    newX: number,
    newY: number,
    callbacks: any,
    isDevMode: boolean = false
  ): { canPass: boolean; newState: GameState } {
    const { setMessage } = callbacks;

    const hasRequiredItems = barrier.requiredItems.every(item =>
      state.inventory.includes(item)
    );

    if (hasRequiredItems) {
      setMessage(GAME_MESSAGES.BARRIER_PASSED('You'));
      const pointChange = isDevMode ? 2 : -2;
      return {
        canPass: true,
        newState: {
          ...state,
          stepsRemaining: state.stepsRemaining + pointChange
        }
      };
    } else {
      const requiredItemsText = barrier.requiredItems.map(item => {
        if (item === 'blueAmulet') return 'blue amulet';
        if (item === 'redAmulet') return 'red amulet';
        return item;
      }).join(' and ');

      if (barrier.requiredItems.includes('blueAmulet')) {
        setMessage(GAME_MESSAGES.BARRIER_BLUE('You'));
      } else if (barrier.requiredItems.includes('redAmulet')) {
        setMessage(GAME_MESSAGES.BARRIER_RED('You'));
      }

      return { canPass: false, newState: state };
    }
  }

  static moveNPCMultiple(
    npc: Agent,
    blocks: any[],
    barriers: any[],
    treasurePots: any[],
    wizards: any[],
    movesCount: number,
    callbacks: any,
    state: GameState
  ): Agent {
    let newNPC = { ...npc };
    
    // Initialize movement history if it doesn't exist
    if (!newNPC.movementHistory) {
      newNPC.movementHistory = [];
    }
    
    for (let i = 0; i < movesCount; i++) {
      if (newNPC.currentMovementIndex >= newNPC.currentPath!.length) {
        // Agent has reached the end of their path - only notify once when they first complete
        if (newNPC.currentMovementIndex === newNPC.currentPath!.length && callbacks.setMessage) {
          callbacks.setMessage(GAME_MESSAGES.AGENT_PATH_COMPLETE(newNPC.id, newNPC.color));
        }
        break;
      }
      
      const prevX = newNPC.x;
      const prevY = newNPC.y;
      const currentMovement = newNPC.currentPath![newNPC.currentMovementIndex];
      const directions: { [key: string]: [number, number] } = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
      const [dx, dy] = directions[currentMovement] || [0, 0];
      const newX = newNPC.x + dx;
      const newY = newNPC.y + dy;

      // Check for wizard interactions first
      const wizardIndex = wizards.findIndex((w: any) => w.x === newX && w.y === newY);
      if (wizardIndex !== -1) {
        const wizard = wizards[wizardIndex];

        // Check if this agent has already interacted with this wizard
        if (!wizard.interactedBy.includes(newNPC.id)) {
          // Agent interacts with wizard but doesn't move to wizard position
          let interactionMessage = '';
          if (wizard.content === 'blueAmulet') {
            interactionMessage = GAME_MESSAGES.BLUE_AMULET(newNPC.id, newNPC.color, EXPERIMENT_TYPE);
          } else if (wizard.content === 'redAmulet') {
            interactionMessage = GAME_MESSAGES.RED_AMULET(newNPC.id, newNPC.color, EXPERIMENT_TYPE);
          } else if (wizard.content === 'nothing') {
            interactionMessage = GAME_MESSAGES.EMPTY_WIZARD(newNPC.id, newNPC.color, EXPERIMENT_TYPE);
          }

          if (interactionMessage && callbacks.setMessage) {
            // Append wizard and agent data for animation
            const animationData = JSON.stringify({
              wizardX: wizard.x,
              wizardY: wizard.y,
              agentId: newNPC.id,
              wizardColor: wizard.color
            });
            const fullMessage = `${interactionMessage}|${animationData}`;
            console.log("GameLogic: Setting message with animation data:", fullMessage);
            callbacks.setMessage(fullMessage);
          }

          // Mark wizard as interacted with by this agent
          wizard.interactedBy.push(newNPC.id);
        }

        // Don't actually move the agent - keep it at original position
        // The visual "move toward" will be handled by CSS animation
        newNPC = { ...newNPC, currentMovementIndex: newNPC.currentMovementIndex + 1 };

        // Add movement to history
        const newHistory = [
          { x: prevX, y: prevY, direction: currentMovement, timestamp: Date.now() },
          ...(newNPC.movementHistory || [])
        ].slice(0, this.TRAIL_LENGTH);
        newNPC.movementHistory = newHistory;

      } else if (this.isValidMove(newX, newY, blocks, wizards)) {
        // Normal movement
        newNPC = { ...newNPC, x: newX, y: newY, currentMovementIndex: newNPC.currentMovementIndex + 1 };
        
        // Add movement to history (previous position with direction)
        const newHistory = [
          { x: prevX, y: prevY, direction: currentMovement, timestamp: Date.now() },
          ...(newNPC.movementHistory || [])
        ].slice(0, this.TRAIL_LENGTH); // Keep last N movements
        
        newNPC.movementHistory = newHistory;
        
        if (treasurePots.some((pot: any) => pot.x === newX && pot.y === newY)) {
          // NPC has reached a treasure chest, mark it as disappeared
          // newNPC.disappeared = true;
          break;
        }
      } else {
        // Invalid move, just increment movement index
        newNPC = { ...newNPC, currentMovementIndex: newNPC.currentMovementIndex + 1 };
      }
    }
    
    newNPC.observesRemaining = Math.max(0, newNPC.observesRemaining - 1);
    return newNPC;
  }

  private static isValidMove(
    x: number,
    y: number,
    blocks: any[],
    wizards: any[]
  ): boolean {
    return x >= 0 && x < 11 && y >= 0 && y < 12 && 
      !blocks.some(b => b.x === x && b.y === y) && 
      !wizards.some(w => w.x === x && w.y === y);
  }

  // Helper function to move all agents in a state
  private static moveAllAgents(
    state: GameState, 
    movesCount: number, 
    callbacks: {
      setMessage: (message: string) => void;
      setActionLog?: (log: any[]) => void;
      setGameWon?: (won: boolean) => void;
    }
  ): GameState {
    const updatedAgents = state.agents.map(agent => {
      // Only move if agent has remaining path
      if (agent.currentPath && agent.currentMovementIndex < agent.currentPath.length) {
        return this.moveNPCMultiple(
          agent,
          state.blocks,
          state.barriers,
          state.treasurePots,
          state.wizards,
          movesCount,
          callbacks,
          state
        );
      }
      return agent;
    });

    return {
      ...state,
      agents: updatedAgents
    };
  }

  private static calculateNewState(
    state: GameState,
    direction: string,
    callbacks: {
      setMessage: (message: string) => void;
      setActionLog?: (log: any[]) => void;
      setGameWon?: (won: boolean) => void;
    },
    currentLevel?: string,
    isDevMode: boolean = false
  ): GameState {
    const { setMessage, setActionLog, setGameWon } = callbacks;
    const [dx, dy] = {
      up: [0, -1],
      down: [0, 1],
      left: [-1, 0],
      right: [1, 0]
    }[direction] || [0, 0];

    const newX = state.player.x + dx;
    const newY = state.player.y + dy;

    // Check for wizard interactions
    const wizardIndex = state.wizards.findIndex(w => w.x === newX && w.y === newY);
    if (wizardIndex !== -1) {
      return this.handleWizardInteraction(state, wizardIndex, callbacks, state.gameSessionId, currentLevel, isDevMode);
    }

    // Check for treasure interactions
    const treasure = state.treasurePots.find(p => p.x === newX && p.y === newY);
    if (treasure) {
      return this.handleTreasureInteraction(state, treasure, newX, newY, callbacks, state.gameSessionId, currentLevel);
    }

    // Handle barrier interaction
    const barrier = state.barriers.find(b => b.x === newX && b.y === newY);
    if (barrier) {
      const { canPass, newState } = this.handleBarrierInteraction(state, barrier, newX, newY, callbacks, isDevMode);
      if (!canPass) return state;

      return {
        ...newState,
        player: { x: newX, y: newY, isPlayer: true, id: 0 }
      };
    }

    // Regular movement - agents do NOT move when player moves
    if (this.isValidMove(newX, newY, state.blocks, state.wizards)) {
      const pointChange = isDevMode ? 3 : -3;
      return {
        ...state,
        player: { x: newX, y: newY, isPlayer: true, id: 0 },
        stepsRemaining: state.stepsRemaining + pointChange
      };
    }

    return state;
  }
}