// User-centric Firebase structure
export interface GameEvent {
  type: 'level_start' | 'player_action' | 'agent_observe' | 'interaction' | 'level_complete' | 'game_end' | 'experiment_complete';
  timestamp: number;
  [key: string]: any; // Additional event-specific data
}

export interface PlayerAction extends GameEvent {
  type: 'player_action';
  action: string;
  playerPos: { x: number; y: number };
}

export interface AgentObserve extends GameEvent {
  type: 'agent_observe';
  agentId: number;
  agentPos: { x: number; y: number };
  playerPos?: { x: number; y: number };
  pathType: string;
}

export interface Interaction extends GameEvent {
  type: 'interaction';
  interactionType: string;
  [key: string]: any;
}

export interface LevelData {
  levelId: string;
  startTime: number;
  endTime?: number;
  status: 'active' | 'completed' | 'failed';
  eventCount: number;
  finalSteps?: number;
  events: { [eventId: string]: GameEvent };
}

export interface UserData {
  sessionId: string;
  startTime: number;
  endTime?: number;
  status: 'active' | 'completed';
  totalEvents: number;
  lastActivity?: number;
  currentLevel?: string;
  finalOutcome?: string;
  finalScore?: number;
  demographics?: {
    prolificId?: string;
    age?: number;
    gender?: string;
    feedback?: string;
    timestamp: number;
  };
  levels: { [levelId: string]: LevelData };
}

export interface UserInfo {
  prolificId: string;
  age: number;
  gender: string;
  feedback?: string; // Keep feedback optional
}

export interface GameSession extends UserData {} // Alias for backward compatibility

export interface FirebaseLoggerConfig {
  enableLogging?: boolean;
  captureGameState?: boolean;
  batchSize?: number;
  flushOnLevelComplete?: boolean;
  flushInterval?: number;
}