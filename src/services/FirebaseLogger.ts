import { ref, push, set, serverTimestamp, update, runTransaction, get } from 'firebase/database';
import { db } from '../config/firebaseConfig';
import { GameEvent, UserInfo, GameSession, FirebaseLoggerConfig } from '../types/firebase';
import { GameState } from '../types/game';

interface BatchedEvent {
  type: GameEvent['type'];
  data: any;
  timestamp: number;
  levelId: string;
}

interface LevelBatch {
  levelId: string;
  events: BatchedEvent[];
  metadata: {
    startTime: number;
    eventCount: number;
    playerActions: number;
    agentObservations: number;
    interactions: number;
  };
}

export class FirebaseLogger {
  private sessionId: string | null = null;
  private config: FirebaseLoggerConfig;
  private sessionStartTime: number = 0;
  private isInitialized = false;
  private currentLevel: string = '';
  
  // Batch management
  private eventBatch: BatchedEvent[] = [];
  private levelBatches: Map<string, LevelBatch> = new Map();
  private batchTimer: NodeJS.Timeout | null = null;
  // Remove the hard-coded MAX_BATCH_SIZE and make it configurable
  private readonly BATCH_TIMEOUT_MS = 30000; // 30 seconds

  constructor(config: FirebaseLoggerConfig = {}) {
    this.config = {
      enableLogging: true,
      captureGameState: false,
      batchSize: 125, // Increased from 20 to 100
      flushOnLevelComplete: true,
      flushInterval: 30000, // 30 seconds
      ...config
    };
  }

  /**
   * Get and increment the global session counter atomically
   */
  private async getAndIncrementCounter(): Promise<number> {
    const counterRef = ref(db, 'counters/sessionCounter');

    try {
      const result = await runTransaction(counterRef, (currentValue) => {
        // If counter doesn't exist, initialize to 1
        if (currentValue === null) {
          return 1;
        }
        // Otherwise increment
        return currentValue + 1;
      });

      return result.snapshot.val() as number;
    } catch (error) {
      console.error('Failed to increment counter:', error);
      // Fallback to timestamp-based counter if transaction fails
      return Date.now() % 10000;
    }
  }

  /**
   * Initialize a new game session with user-centric structure
   * Returns both sessionId and sessionNumber
   */
  async initializeSession(): Promise<{ sessionId: string; sessionNumber: number }> {
    if (!this.config.enableLogging) {
      return { sessionId: 'logging-disabled', sessionNumber: 1 };
    }

    try {
      // Get the incremented counter for this session
      const sessionNumber = await this.getAndIncrementCounter();

      this.sessionId = `u${Date.now()}`; // User-centric ID
      this.sessionStartTime = Date.now();

      // Initialize user with basic info including session number
      const userData = {
        sessionId: this.sessionId,
        sessionNumber: sessionNumber,
        startTime: this.sessionStartTime,
        status: 'active',
        totalEvents: 0,
        levels: {} // This will contain all level data
      };

      await set(ref(db, `users/${this.sessionId}`), userData);
      this.isInitialized = true;

      console.log(`🔥 Firebase user session: ${this.sessionId}, Session #${sessionNumber}`);
      return { sessionId: this.sessionId, sessionNumber: sessionNumber };
    } catch (error) {
      console.error('Failed to initialize Firebase session:', error);
      this.sessionId = 'error-' + Date.now();
      return { sessionId: this.sessionId, sessionNumber: 1 };
    }
  }

  /**
   * Log consent preference (recontact)
   */
  async logConsentPreference(preference: { doNotRecontact: boolean; timestamp: string }): Promise<void> {
    if (!this.config.enableLogging) return;

    try {
      // Store consent preference in a simple consents collection
      const consentRef = push(ref(db, 'consents'));
      await set(consentRef, {
        doNotRecontact: preference.doNotRecontact,
        timestamp: preference.timestamp,
        sessionId: this.sessionId || 'pending'
      });
    } catch (error) {
      console.error('Failed to log consent preference:', error);
    }
  }

  /**
   * Add user demographic info
   */
  async addUserInfo(userInfo: UserInfo): Promise<void> {
    if (!this.sessionId || !this.config.enableLogging) {
      throw new Error('Firebase logging is not initialized. Cannot save user info.');
    }

    try {
      const demographicsData: any = {
        prolificId: userInfo.prolificId,
        age: userInfo.age,
        gender: userInfo.gender,
        timestamp: Date.now()
      };
      
      // Only include feedback if it exists (Firebase doesn't handle undefined well)
      if (userInfo.feedback !== undefined && userInfo.feedback !== null) {
        demographicsData.feedback = userInfo.feedback;
      }

      console.log('Saving user demographics to Firebase:', {
        sessionId: this.sessionId,
        data: demographicsData
      });

      // Verify session still exists before updating
      const sessionRef = ref(db, `users/${this.sessionId}`);
      const snapshot = await get(sessionRef);

      if (!snapshot.exists()) {
        throw new Error(`Session ${this.sessionId} no longer exists in database`);
      }

      await update(sessionRef, {
        demographics: demographicsData
      });

      console.log('✅ User demographics saved to Firebase successfully');

      // Verify the data was actually written
      const verifySnapshot = await get(ref(db, `users/${this.sessionId}/demographics`));
      if (!verifySnapshot.exists()) {
        throw new Error('Demographics data verification failed - data not found after write');
      }

      console.log('✅ Demographics data verified in Firebase');
    } catch (error) {
      console.error('❌ Failed to add user info:', error);
      // Re-throw the error so the caller knows it failed
      throw new Error(`Failed to save user info: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Log events with batching - events are queued and written in batches
   */
  async logEvent(
    type: GameEvent['type'], 
    data: any, 
    levelId: string
  ): Promise<void> {
    if (!this.sessionId || !this.config.enableLogging) return;

    try {
      this.currentLevel = levelId;
      
      const batchedEvent: BatchedEvent = {
        type: type,
        data: data,
        timestamp: Date.now(),
        levelId: levelId
      };

      // Add to main event batch
      this.eventBatch.push(batchedEvent);

      // Add to level-specific batch
      this.addToLevelBatch(batchedEvent);

      // Check if we need to flush - use configurable batch size instead of hard-coded MAX_BATCH_SIZE
      if (this.eventBatch.length >= (this.config.batchSize || 50)) {
        await this.flushBatch('max_size_reached');
      } else if (!this.batchTimer) {
        // Start timer for periodic flush
        this.startBatchTimer();
      }
      
    } catch (error) {
      console.error(`Failed to queue ${type}:`, error);
    }
  }

  /**
   * Add event to level-specific batch for tracking
   */
  private addToLevelBatch(event: BatchedEvent): void {
    let levelBatch = this.levelBatches.get(event.levelId);
    
    if (!levelBatch) {
      levelBatch = {
        levelId: event.levelId,
        events: [],
        metadata: {
          startTime: Date.now(),
          eventCount: 0,
          playerActions: 0,
          agentObservations: 0,
          interactions: 0
        }
      };
      this.levelBatches.set(event.levelId, levelBatch);
    }

    levelBatch.events.push(event);
    levelBatch.metadata.eventCount++;

    // Update counters based on event type
    switch (event.type) {
      case 'player_action':
        levelBatch.metadata.playerActions++;
        break;
      case 'agent_observe':
        levelBatch.metadata.agentObservations++;
        break;
      case 'interaction':
        levelBatch.metadata.interactions++;
        break;
    }
  }

  /**
   * Start the batch timer for periodic flushing
   */
  private startBatchTimer(): void {
    if (this.batchTimer) return;
    
    this.batchTimer = setTimeout(async () => {
      await this.flushBatch('timer_expired');
      this.batchTimer = null;
    }, this.BATCH_TIMEOUT_MS);
  }

  /**
   * Flush all batched events to Firebase
   */
  private async flushBatch(reason: string = 'manual'): Promise<void> {
    if (this.eventBatch.length === 0 || !this.sessionId) return;

    try {
      console.log(`Flushing ${this.eventBatch.length} events (reason: ${reason})`);

      // Prepare batch updates
      const updates: { [key: string]: any } = {};
      const eventsByLevel: { [levelId: string]: BatchedEvent[] } = {};

      // Group events by level
      this.eventBatch.forEach(event => {
        if (!eventsByLevel[event.levelId]) {
          eventsByLevel[event.levelId] = [];
        }
        eventsByLevel[event.levelId].push(event);
      });

      // Create batch updates for each level
      for (const [levelId, events] of Object.entries(eventsByLevel)) {
        const levelBatch = this.levelBatches.get(levelId);
        
        // Create a batch object for this level
        const batchId = `batch_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const batchPath = `users/${this.sessionId}/levels/${levelId}/batches/${batchId}`;
        
        updates[batchPath] = {
          events: events,
          metadata: levelBatch?.metadata || {
            startTime: Date.now(),
            eventCount: events.length,
            playerActions: events.filter(e => e.type === 'player_action').length,
            agentObservations: events.filter(e => e.type === 'agent_observe').length,
            interactions: events.filter(e => e.type === 'interaction').length
          },
          batchInfo: {
            batchId: batchId,
            flushReason: reason,
            flushTime: Date.now(),
            eventCount: events.length
          }
        };

        // Update level metadata
        const levelMetadataPath = `users/${this.sessionId}/levels/${levelId}/summary`;
        updates[levelMetadataPath] = {
          lastEvent: Date.now(),
          totalEvents: (levelBatch?.metadata.eventCount || 0),
          lastBatchId: batchId
        };
      }

      // Update user-level metadata
      updates[`users/${this.sessionId}/lastActivity`] = Date.now();
      updates[`users/${this.sessionId}/currentLevel`] = this.currentLevel;
      updates[`users/${this.sessionId}/totalBatchedEvents`] = this.eventBatch.length;

      // Perform the batch update
      await update(ref(db), updates);

      // Clear the batch
      this.eventBatch = [];
      
      // Clear batch timer
      if (this.batchTimer) {
        clearTimeout(this.batchTimer);
        this.batchTimer = null;
      }

      console.log('Batch flush completed successfully');
      
    } catch (error) {
      console.error('Failed to flush batch:', error);
      // Don't clear the batch on error - retry later
    }
  }

  /**
   * Log player action with coordinates
   */
  async logPlayerAction(action: string, levelId: string, pos: { x: number; y: number }): Promise<void> {
    await this.logEvent('player_action', { 
      action: action,
      playerPos: pos
    }, levelId);
  }

  /**
   * Log agent observation with both player and agent coordinates
   */
  async logAgentObserve(agentId: number, levelId: string, agentPos: { x: number; y: number }, pathType: string, playerPos?: { x: number; y: number }): Promise<void> {
    await this.logEvent('agent_observe', { 
      agentId: agentId,
      agentPos: agentPos,
      playerPos: playerPos,
      pathType: pathType
    }, levelId);
  }

  /**
   * Log interactions with coordinates
   */
  async logInteraction(type: string, details: any, levelId: string): Promise<void> {
    await this.logEvent('interaction', { 
      interactionType: type,
      ...details
    }, levelId);
  }

  /**
   * Log level start
   */
  async logGameStart(levelId: string): Promise<void> {
    // Initialize level structure
    await set(ref(db, `users/${this.sessionId}/levels/${levelId}`), {
      levelId: levelId,
      startTime: Date.now(),
      status: 'active',
      eventCount: 0,
      events: {}
    });

    await this.logEvent('level_start', { 
      level: levelId,
      startTime: Date.now()
    }, levelId);
  }

  /**
   * Log level change
   */
  async logLevelChange(fromLevel: string, toLevel: string): Promise<void> {
    // Mark previous level as complete
    if (fromLevel) {
      await update(ref(db, `users/${this.sessionId}/levels/${fromLevel}`), {
        status: 'completed',
        endTime: Date.now()
      });
    }

    // Start new level
    await this.logGameStart(toLevel);
  }

  /**
   * Log level completion and flush all batched data for this level
   */
  async logLevelComplete(levelId: string, success: boolean, stepsUsed: number): Promise<void> {
    // Add the level completion event to batch
    await this.logEvent('level_complete', { 
      success: success,
      stepsUsed: stepsUsed,
      duration: Date.now() - this.sessionStartTime
    }, levelId);

    // Force flush batch for level completion
    if (this.config.flushOnLevelComplete) {
      await this.flushBatch('level_complete');
    }

    // Update level status with final summary
    const levelBatch = this.levelBatches.get(levelId);
    await update(ref(db, `users/${this.sessionId}/levels/${levelId}`), {
      status: success ? 'completed' : 'failed',
      endTime: Date.now(),
      finalSteps: stepsUsed,
      summary: levelBatch?.metadata || {
        eventCount: 0,
        playerActions: 0,
        agentObservations: 0,
        interactions: 0
      }
    });

    // Clear this level's batch data
    this.levelBatches.delete(levelId);
  }

  /**
   * Log game end and flush all remaining batched data
   */
  async logGameEnd(outcome: string, levelId: string, finalScore: number): Promise<void> {
    await this.logEvent('game_end', { 
      outcome: outcome,
      finalScore: finalScore,
      totalDuration: Date.now() - this.sessionStartTime
    }, levelId);
    
    // Force flush all remaining batched data
    await this.flushBatch('game_end');
    
    // Mark user session as complete
    await update(ref(db, `users/${this.sessionId}`), {
      status: 'completed',
      endTime: Date.now(),
      finalOutcome: outcome,
      finalScore: finalScore,
      totalLevelsCompleted: this.levelBatches.size
    });

    // Clean up
    this.levelBatches.clear();
    if (this.batchTimer) {
      clearTimeout(this.batchTimer);
      this.batchTimer = null;
    }
  }

  /**
   * Manually flush any pending batches (useful for debugging or force sync)
   */
  async flushPendingBatches(): Promise<void> {
    await this.flushBatch('manual_flush');
  }

  /**
   * Get batch status for debugging
   */
  getBatchStatus(): {
    pendingEvents: number;
    activeLevels: string[];
    batchTimerActive: boolean;
  } {
    return {
      pendingEvents: this.eventBatch.length,
      activeLevels: Array.from(this.levelBatches.keys()),
      batchTimerActive: this.batchTimer !== null
    };
  }

  /**
   * Get current session ID
   */
  getSessionId(): string | null {
    return this.sessionId;
  }

  /**
   * Check if logging is enabled and initialized
   */
  isLoggingEnabled(): boolean {
    return this.config.enableLogging === true && this.isInitialized;
  }

  /**
   * Helper methods for counting events
   */
  private async getLevelEventCount(levelId: string): Promise<number> {
    // Simplified - return 0 for now
    return 0;
  }

  private async getTotalEventCount(): Promise<number> {
    // Simplified - return 0 for now
    return 0;
  }
}

// Singleton instance for global use
export const firebaseLogger = new FirebaseLogger();