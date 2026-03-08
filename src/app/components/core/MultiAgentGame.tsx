"use client";
import React, { useState, useCallback, useMemo, useEffect } from "react";
import { GameLogic } from "@/services/GameLogic";
import { firebaseLogger } from "@/services/FirebaseLogger";
import GameGrid from "./GameGrid";
import GameInfo from "./GameInfo";
import ActionButtons from "../ui/ActionButtons";
import Inventory from "../ui/Inventory";
import { loadLevel, levels } from "@/data/levels";
import DebugPasswordModal from "../debug/DebugPasswordModal";
import DebugMenuModal from "../debug/DebugMenuModal";
import DebugPanel from "../debug/DebugPanel";
import UserDataModal from "../modals/UserDataModal";
import GameIntroduction from "../game-flow/GameIntroduction";
import ComprehensionCheck from "../game-flow/ComprehensionCheck";
import ConsentForm from "../game-flow/ConsentForm";
import Debrief from "../game-flow/Debrief";
import useLevelSequencer, {
  LevelCompleteInfo,
} from "../game-flow/LevelSequencer";

interface StyledMessageProps {
  message: string | null;
}

const StyledMessage: React.FC<StyledMessageProps> = ({ message }) => {
  const [isInitialRender, setIsInitialRender] = useState(true);
  const [interactionCount, setInteractionCount] = useState(0);

  useEffect(() => {
    if (isInitialRender) {
      setIsInitialRender(false);
      return;
    }

    // Trigger animation for wizard interactions and agent interactions
    if (
      message &&
      (message.includes("wizard") ||
        message.includes("already interact") ||
        message.includes("P1") ||
        message.includes("P2") ||
        message.includes("<img") ||
        message.includes("YOU WON!") ||
        message.includes("Wrong Treasure!") ||
        message.includes("No more observations"))
    ) {
      setInteractionCount((prev) => prev + 1);
    }
  }, [message]);

  const displayMessage = message ? message.split("|")[0] : null;

  return (
    <div
      key={interactionCount}
      className={`bg-white p-6 rounded-lg shadow-md mb-6 max-w-3xl mx-auto min-h-[100px] flex items-center justify-center ${!isInitialRender ? "animate-notification" : ""}`}
    >
      <p className="text-3xl text-center">
        {displayMessage
          ? displayMessage
              .split(/(<span.*?<\/span>|<img[^>]*>)/)
              .map((part: string, index: number) => {
                if (part.startsWith("<img")) {
                  const src = part.match(/src="([^"]*)"/)?.[1];
                  const alt = part.match(/alt="([^"]*)"/)?.[1];
                  return (
                    <img
                      key={index}
                      src={src}
                      alt={alt || ""}
                      style={{
                        display: "inline",
                        width: "32px",
                        height: "40px",
                        verticalAlign: "middle",
                        marginRight: "6px",
                      }}
                    />
                  );
                }
                if (part.startsWith("<span")) {
                  const color = part.match(/color: (\w+)/)?.[1];
                  const isBold = part.includes("font-weight: bold");
                  const content = part.replace(/<\/?span[^>]*>/g, "");
                  return (
                    <span
                      key={index}
                      style={{
                        color: color || "inherit",
                        fontWeight: isBold ? "bold" : "normal",
                      }}
                    >
                      {content}
                    </span>
                  );
                }
                return part;
              })
          : null}
      </p>
    </div>
  );
};

const styles = `
  @keyframes notification {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
  }
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }
  @keyframes shrink {
    from {
      transform: scale(1);
      opacity: 1;
    }
    to {
      transform: scale(0);
      opacity: 0.3;
    }
  }
  .animate-notification {
    animation: notification 0.3s ease-out;
  }
  .animate-fadeIn {
    animation: fadeIn 0.3s ease-out;
  }
`;

// Configuration for agent movement behavior
const AGENT_MOVEMENT_CONFIG = {
  ALWAYS_MOVING: true, // Set to false to revert to original behavior
  MOVES_ON_OBSERVE: 1, // Number of moves when player observes (changed from 2 to 1)
  MOVES_ON_PLAYER_ACTION: 1, // Number of moves when player acts (original: 0)
};

// Helper function to extract experiment folder from level ID
function getExperimentFolder(levelId: string): string {
  if (levelId.includes('_exp4')) return 'exp4';
  if (levelId.includes('_true')) return 'exp3_true';
  if (levelId.startsWith('sm')) return 'exp3';
  if (levelId.startsWith('s') && !levelId.startsWith('sm')) return 'exp2';
  if (levelId.startsWith('mod_')) return 'mod';
  return 'other';
}

// Helper function to get all experiment folders
function getExperimentFolders(): string[] {
  const folders = new Set<string>();
  Object.keys(levels).forEach(levelId => {
    folders.add(getExperimentFolder(levelId));
  });
  return Array.from(folders).sort();
}

export function MultiAgentGame() {
  const [currentLevel, setCurrentLevel] = useState("s111_1"); // Initialize with variant
  const [state, setState] = useState(() =>
    GameLogic.initializeGame(loadLevel("s111")),
  );
  const [message, setMessage] = useState<string>("");
  const [actionLog, setActionLog] = useState<any[]>([]);
  const [gameWon, setGameWon] = useState(false);
  const [showDebugPasswordModal, setShowDebugPasswordModal] = useState(false);
  const [showDebugMenuModal, setShowDebugMenuModal] = useState(false);
  const [isDebugMode, setIsDebugMode] = useState(false);
  const [isDevMode, setIsDevMode] = useState(false);
  const [stateLogs, setStateLogs] = useState<
    Array<{
      timestamp: number;
      type: string;
      details: string;
    }>
  >([]);
  const [actionsDisabled, setActionsDisabled] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [lastObserveTime, setLastObserveTime] = useState<number | null>(null);
  const [observeCooldownRemaining, setObserveCooldownRemaining] = useState(0);
  const [lastMoveTime, setLastMoveTime] = useState<number | null>(null);
  const [moveCooldownRemaining, setMoveCooldownRemaining] = useState(0);

  // Debug controls for experiment folder and scenario selection
  const [selectedExperimentFolder, setSelectedExperimentFolder] = useState<string>("exp4");
  const [selectedScenario, setSelectedScenario] = useState<number>(1);

  // Observation period state
  const [isObservationPeriod, setIsObservationPeriod] = useState(false);
  const [observationCountdown, setObservationCountdown] = useState(10);

  // Wizard interaction animation state
  const [activeInteraction, setActiveInteraction] = useState<{
    wizardX: number;
    wizardY: number;
    agentId: number | null;
    wizardColor: string;
    timestamp: number;
  } | null>(null);

  // Cooldown durations in milliseconds
  const OBSERVE_COOLDOWN_MS = 333; // 1/3 second
  const MOVE_COOLDOWN_MS = 200; // 0.2 seconds

  // Game flow states
  const [gameFlowState, setGameFlowState] = useState<
    | "consent"
    | "intro"
    | "comprehension"
    | "game"
    | "tutorialComplete"
    | "trialComplete"
    | "userInfo"
    | "thankYou"
  >("consent");
  const [comprehensionActionLog, setComprehensionActionLog] = useState<any[]>(
    [],
  );

  // Sequencer state
  const [sessionNumber, setSessionNumber] = useState<number>(1); // Default to 1 for now
  const [lastTrialPoints, setLastTrialPoints] = useState(0);
  const [lastTrialDisplayedPoints, setLastTrialDisplayedPoints] = useState(0);

  const addStateLog = useCallback((type: string, details: string) => {
    setStateLogs((prev) => [
      ...prev,
      {
        timestamp: Date.now(),
        type,
        details,
      },
    ]);
  }, []);

  // Handle observation period countdown (only in non-debug mode)
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (isObservationPeriod && observationCountdown > 0 && !isDebugMode) {
      interval = setInterval(() => {
        setObservationCountdown((prev) => {
          if (prev <= 1) {
            setIsObservationPeriod(false);
            setActionsDisabled(false);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isObservationPeriod, observationCountdown, isDebugMode]);

  // Handle observe cooldown timer
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (lastObserveTime !== null) {
      interval = setInterval(() => {
        const now = Date.now();
        const timeSinceLastObserve = now - lastObserveTime;
        const remaining = Math.max(
          0,
          OBSERVE_COOLDOWN_MS - timeSinceLastObserve,
        );

        setObserveCooldownRemaining(Math.ceil(remaining / 1000)); // Convert to seconds

        if (remaining <= 0) {
          setLastObserveTime(null);
        }
      }, 100); // Update every 100ms for smooth countdown
    } else {
      setObserveCooldownRemaining(0);
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [lastObserveTime, OBSERVE_COOLDOWN_MS]);

  // Handle move cooldown timer
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (lastMoveTime !== null) {
      interval = setInterval(() => {
        const now = Date.now();
        const timeSinceLastMove = now - lastMoveTime;
        const remaining = Math.max(
          0,
          MOVE_COOLDOWN_MS - timeSinceLastMove,
        );

        setMoveCooldownRemaining(Math.ceil(remaining / 1000)); // Convert to seconds

        if (remaining <= 0) {
          setLastMoveTime(null);
        }
      }, 100); // Update every 100ms for smooth countdown
    } else {
      setMoveCooldownRemaining(0);
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [lastMoveTime, MOVE_COOLDOWN_MS]);

  // Clear observation period immediately when debug mode is enabled
  useEffect(() => {
    if (isDebugMode && isObservationPeriod) {
      setIsObservationPeriod(false);
      setActionsDisabled(false);
    }
  }, [isDebugMode, isObservationPeriod]);

  // Game flow handlers
  const handleConsentComplete = useCallback(() => {
    setGameFlowState("intro");
  }, []);

  const handleIntroComplete = useCallback(() => {
    setGameFlowState("comprehension");
  }, []);

  const handleComprehensionComplete = useCallback((actionLog: any[]) => {
    setComprehensionActionLog(actionLog);
    setGameFlowState("game");
  }, []);

  const handleComprehensionFailure = useCallback((actionLog: any[]) => {
    setComprehensionActionLog(actionLog);
    // Stay in comprehension state to retry
  }, []);

  const handleReturnToInstructions = useCallback(() => {
    setGameFlowState("intro");
  }, []);

  // Level sequencer handlers
  const handleLevelChange = useCallback(
    (
      levelInfo: { level: string; pathType: string },
      levelIndex: number,
      isTutorial: boolean,
    ) => {
      // Don't allow level sequencer to change levels in debug mode
      if (isDebugMode) {
        return;
      }

      // Construct full level ID with variant for Firebase (e.g., 's211_1' or 's211_2')
      const pathNumber = levelInfo.pathType.replace("experienced", ""); // Extract '1' or '2'
      const fullLevelId = `${levelInfo.level}_${pathNumber}`;

      // Set base level ID for display/loading
      setCurrentLevel(fullLevelId);

      // Strip path suffix for loading level data (new_s111_1 -> new_s111)
      const baseLevelId = levelInfo.level.replace(/_[12]$/, "");
      const initialState = GameLogic.initializeGame(loadLevel(baseLevelId));

      // Map pathType to the correct experienced path
      // pathType is already "experienced1" or "experienced2" from LevelSequencer
      const experiencedPathKey = levelInfo.pathType;

      // Update agents with the correct path based on pathType
      const updatedState = {
        ...initialState,
        agents: initialState.agents.map((agent) => {
          const pathData =
            agent.movements[experiencedPathKey as keyof typeof agent.movements];
          if (pathData) {
            return {
              ...agent,
              selectedPath: experiencedPathKey,
              currentPath: pathData.path,
              type: pathData.type,
              currentMovementIndex: 0,
            };
          }
          return agent;
        }),
      };

      setState(updatedState);
      setGameWon(false);
      setMessage("");
      setLastObserveTime(null); // Reset observe cooldown on level change
      setLastMoveTime(null); // Reset move cooldown on level change

      // Actions enabled/disabled will be handled by observation effect when game starts
      if (isDebugMode) {
        setActionsDisabled(false); // Debug mode starts with actions enabled
      }

      // Log level change to Firebase with full level ID including variant
      if (firebaseLogger.isLoggingEnabled()) {
        firebaseLogger
          .logLevelChange(currentLevel, fullLevelId)
          .catch((error) =>
            console.error("Failed to log level change:", error),
          );
      }
    },
    [currentLevel, isDebugMode],
  );

  const handleTutorialComplete = useCallback(() => {
    setGameFlowState("tutorialComplete");
  }, []);

  const handleTrialComplete = useCallback(
    (pointsEarned: number, displayedPointsEarned: number) => {
      setLastTrialPoints(pointsEarned);
      setLastTrialDisplayedPoints(displayedPointsEarned);
      setGameFlowState("trialComplete");
    },
    [],
  );

  // Start observation period when entering the game screen or when level changes during gameplay
  useEffect(() => {
    if (!isDebugMode && gameFlowState === "game") {
      setActionsDisabled(true);
      setIsObservationPeriod(true);
      setObservationCountdown(10);
    }
  }, [gameFlowState, currentLevel, isDebugMode]);

  const handleExperimentComplete = useCallback(async () => {
    setGameFlowState("userInfo");
  }, []);

  const handleDebugLevelChange = useCallback(
    (level: string) => {
      const previousLevel = currentLevel;
      setCurrentLevel(level);
      const initialState = GameLogic.initializeGame(
        loadLevel(level),
        false,
        isDevMode,
      );

      // Apply selected scenario to all agents
      const experiencedPathKey = `experienced${selectedScenario}`;
      const updatedState = {
        ...initialState,
        agents: initialState.agents.map((agent) => {
          const pathData =
            agent.movements[experiencedPathKey as keyof typeof agent.movements];
          if (pathData) {
            return {
              ...agent,
              selectedPath: experiencedPathKey,
              currentPath: pathData.path,
              type: pathData.type,
              currentMovementIndex: 0,
            };
          }
          return agent;
        }),
      };

      setState(updatedState);

      // Reset game state when changing levels in debug mode
      setGameWon(false);
      setActionsDisabled(false);
      setMessage("");
      setLastObserveTime(null); // Reset observe cooldown on debug level change
      setLastMoveTime(null); // Reset move cooldown on debug level change

      // Log level change to Firebase
      if (firebaseLogger.isLoggingEnabled() && previousLevel !== level) {
        firebaseLogger
          .logLevelChange(previousLevel, level)
          .catch((error) =>
            console.error("Failed to log level change:", error),
          );
      }
    },
    [currentLevel, isDevMode, selectedScenario],
  );

  const handleScenarioChange = useCallback(
    (scenario: number) => {
      setSelectedScenario(scenario);

      // Re-apply the scenario to the current level
      const experiencedPathKey = `experienced${scenario}`;
      const updatedState = {
        ...state,
        agents: state.agents.map((agent) => {
          const pathData =
            agent.movements[experiencedPathKey as keyof typeof agent.movements];
          if (pathData) {
            return {
              ...agent,
              selectedPath: experiencedPathKey,
              currentPath: pathData.path,
              type: pathData.type,
              currentMovementIndex: 0,
            };
          }
          return agent;
        }),
      };

      setState(updatedState);
      setMessage("");
    },
    [state],
  );

  const getNextLevel = useCallback((currentLevelId: string) => {
    const levelIds = Object.keys(levels);
    const currentIndex = levelIds.indexOf(currentLevelId);
    if (currentIndex < levelIds.length - 1) {
      return levelIds[currentIndex + 1];
    }
    return null; // Return null if we're at the last level
  }, []);

  // Enhanced setMessage that tracks wizard interactions for animation
  const setMessageWithInteraction = useCallback((msg: string) => {
    console.log("setMessageWithInteraction called with:", msg);
    setMessage(msg);

    // Check if message contains animation data (format: "message|timestamp|{json}" or "message|{json}")
    if (msg && msg.includes("|")) {
      console.log("Message contains pipe character, splitting...");
      const messageParts = msg.split("|");
      console.log("Message parts:", messageParts);

      // Try to find the JSON part - it should be the last part that starts with "{"
      let animationData = null;
      for (let i = messageParts.length - 1; i >= 0; i--) {
        const part = messageParts[i].trim();
        if (part.startsWith("{")) {
          try {
            animationData = JSON.parse(part);
            console.log("Found and parsed animation data from part", i, ":", animationData);
            break;
          } catch (e) {
            console.log("Failed to parse part", i, "as JSON:", part);
          }
        }
      }

      if (animationData && animationData.wizardX !== undefined && animationData.wizardY !== undefined && animationData.agentId !== undefined) {
        console.log("Setting active interaction:", animationData);
        setActiveInteraction({
          wizardX: animationData.wizardX,
          wizardY: animationData.wizardY,
          agentId: animationData.agentId,
          wizardColor: animationData.wizardColor || 'blue',
          timestamp: Date.now(),
        });

        // Clear interaction after animation duration (made faster: 600ms instead of 800ms)
        setTimeout(() => {
          console.log("Clearing active interaction");
          setActiveInteraction(null);
        }, 600);
      } else {
        console.log("No valid animation data found or missing required fields");
      }
    } else {
      console.log("Message does not contain pipe character or is empty");
    }
  }, []);

  const handleMove = useCallback(
    (direction: string) => {
      // Check if cooldown is still active
      if (lastMoveTime !== null) {
        const now = Date.now();
        const timeSinceLastMove = now - lastMoveTime;
        if (timeSinceLastMove < MOVE_COOLDOWN_MS) {
          return; // Still in cooldown
        }
      }

      if (isDebugMode) addStateLog("Move", `Direction: ${direction}`);
      // Set move cooldown timestamp
      setLastMoveTime(Date.now());

      const agentMovesPerPlayerAction = AGENT_MOVEMENT_CONFIG.ALWAYS_MOVING
        ? AGENT_MOVEMENT_CONFIG.MOVES_ON_PLAYER_ACTION
        : 0; // Original behavior

      const newState = GameLogic.handleMove(
        direction,
        state,
        {
          setMessage: setMessageWithInteraction,
          setActionLog,
          setGameWon: (won: boolean) => {
            setGameWon(won);
            if (won) {
              setActionsDisabled(true); // Disable actions

              // Handle level completion through sequencer (only if not in debug mode)
              setTimeout(() => {
                if (!isDebugMode) {
                  levelSequencer.handleLevelComplete({
                    won: true,
                    remainingSteps: state.stepsRemaining || 0,
                  });
                }
              }, 3000);
            }
          },
        },
        state.gameSessionId,
        currentLevel, // Pass current level for Firebase logging
        agentMovesPerPlayerAction,
        isDevMode,
      );
      setState(newState);
    },
    [
      state,
      isDebugMode,
      isDevMode,
      addStateLog,
      currentLevel,
      getNextLevel,
      handleLevelChange,
      setMessageWithInteraction,
      lastMoveTime,
      MOVE_COOLDOWN_MS,
    ],
  );

  const handleObserve = useCallback(
    (agentId: number) => {
      // Check if cooldown is still active
      if (lastObserveTime !== null) {
        const now = Date.now();
        const timeSinceLastObserve = now - lastObserveTime;
        if (timeSinceLastObserve < OBSERVE_COOLDOWN_MS) {
          return; // Still in cooldown
        }
      }

      if (isDebugMode) addStateLog("Observe", `Agent ${agentId}`);
      // Set observe cooldown timestamp
      setLastObserveTime(Date.now());

      const movesPerObservation = AGENT_MOVEMENT_CONFIG.ALWAYS_MOVING
        ? AGENT_MOVEMENT_CONFIG.MOVES_ON_OBSERVE
        : 1; // Original behavior

      const newState = GameLogic.handleObserve(
        agentId,
        state,
        { setMessage: setMessageWithInteraction, setState },
        state.gameSessionId,
        movesPerObservation,
        currentLevel, // Pass current level for Firebase logging
        isDevMode,
      );
      setState(newState);
    },
    [
      state,
      isDebugMode,
      isDevMode,
      addStateLog,
      currentLevel,
      lastObserveTime,
      OBSERVE_COOLDOWN_MS,
      setMessageWithInteraction,
    ],
  );

  // Initialize Firebase session on component mount
  useEffect(() => {
    const initializeSession = async () => {
      try {
        const { sessionId, sessionNumber } =
          await firebaseLogger.initializeSession();
        setSessionId(sessionId);
        setSessionNumber(sessionNumber);

        // Log game start
        if (firebaseLogger.isLoggingEnabled()) {
          await firebaseLogger.logGameStart(currentLevel);
        }

        // UserDataModal will be shown after experiment completion

        console.log(
          "Game session initialized:",
          sessionId,
          "Session #",
          sessionNumber,
        );
      } catch (error) {
        console.error("Failed to initialize Firebase session:", error);
      }
    };

    initializeSession();
  }, []); // Only run once on mount

  const memoizedAgentInfo = useMemo(
    () =>
      state.agents
        .slice() // Create a copy of the array to avoid mutating the original
        .sort((a, b) => a.id - b.id) // Sort by ID (ascending)
        .map((agent) => ({
          id: agent.id,
          type: agent.type || "Expert",
          color: agent.color,
        })),
    [state.agents],
  );

  const handleDebugPassword = (password: string) => {
    if (password === "debug123") {
      // Replace with your desired password
      setShowDebugPasswordModal(false);
      setShowDebugMenuModal(true);
    } else {
      alert("Incorrect password");
    }
  };

  const handleViewLevels = () => {
    setIsDebugMode(true);
    setShowDebugMenuModal(false);
    setGameFlowState("game");
  };

  const handleViewUserData = () => {
    setShowDebugMenuModal(false);
    setGameFlowState("userInfo");
  };

  const handlePathChange = useCallback(
    (agentId: number, pathType: string) => {
      if (isDebugMode)
        addStateLog("Path Change", `Agent ${agentId} -> ${pathType}`);
      setState((prevState) => ({
        ...prevState,
        agents: prevState.agents.map((agent) =>
          agent.id === agentId
            ? {
                ...agent,
                selectedPath: pathType,
                type: agent.movements[pathType as keyof typeof agent.movements]
                  .type,
                currentPath:
                  agent.movements[pathType as keyof typeof agent.movements]
                    .path,
              }
            : agent,
        ),
      }));
    },
    [isDebugMode, addStateLog],
  );

  const handleReset = useCallback(() => {
    if (isDebugMode) {
      addStateLog("Reset", "Game state reset to initial");

      // First get the current path selections from the agents
      const currentPathSelections = state.agents.map((agent) => ({
        id: agent.id,
        selectedPath: agent.selectedPath,
      }));

      // Initialize a fresh game state from the level data
      const freshState = GameLogic.initializeGame(
        loadLevel(currentLevel),
        false,
        isDevMode,
      );

      // Apply the current path selections to the fresh state
      const newState = {
        ...freshState,
        agents: freshState.agents.map((agent) => {
          const savedSelection = currentPathSelections.find(
            (a) => a.id === agent.id,
          );
          if (savedSelection) {
            return {
              ...agent,
              selectedPath: savedSelection.selectedPath,
              currentPath:
                agent.movements[
                  savedSelection.selectedPath as keyof typeof agent.movements
                ].path,
              type: agent.movements[
                savedSelection.selectedPath as keyof typeof agent.movements
              ].type,
            };
          }
          return agent;
        }),
      };

      setState(newState);
      setMessage("");
      setActionLog([]);
      setGameWon(false);
      setActionsDisabled(false);
      setLastObserveTime(null); // Reset observe cooldown on reset
      setLastMoveTime(null); // Reset move cooldown on reset
    }
  }, [isDebugMode, isDevMode, addStateLog, currentLevel, state.agents]);

  // Generate a session ID for the comprehension check if needed
  const gameSessionId = sessionId || `session_${Date.now()}`;

  // Initialize the level sequencer (always call hook, but disable functionality in debug mode)
  const levelSequencer = useLevelSequencer({
    gameSessionId,
    sessionNumber,
    onLevelChange: handleLevelChange,
    onTutorialComplete: handleTutorialComplete,
    onTrialComplete: handleTrialComplete,
    onExperimentComplete: handleExperimentComplete,
  });

  // Use the sequencer's totals (which correctly exclude tutorial levels)
  const totalPoints = levelSequencer.totalPoints;
  const totalDisplayedPoints = levelSequencer.totalDisplayedPoints;

  // Log experiment completion to Firebase when reaching userInfo screen
  useEffect(() => {
    if (gameFlowState === "userInfo" && !isDebugMode) {
      const logExperimentComplete = async () => {
        try {
          if (firebaseLogger.isLoggingEnabled()) {
            await firebaseLogger.logEvent(
              "experiment_complete",
              {
                totalPointsActual: totalPoints, // Actual total (excludes tutorials, can be negative)
                totalPointsDisplayed: totalDisplayedPoints, // Displayed total (excludes tutorials, clamped to 0+)
                completedAt: new Date().toISOString(),
              },
              "experiment",
            );
          }
        } catch (error) {
          console.error("Error logging experiment complete:", error);
        }
      };
      logExperimentComplete();
    }
  }, [gameFlowState, isDebugMode, totalPoints, totalDisplayedPoints]);

  // Handle different flow states (skip consent/intro/comprehension in debug mode)
  if (!isDebugMode && gameFlowState === "consent") {
    return <ConsentForm onConsent={handleConsentComplete} />;
  }

  if (!isDebugMode && gameFlowState === "intro") {
    return (
      <>
        <GameIntroduction
          onComplete={handleIntroComplete}
          onDebugMode={() => setShowDebugMenuModal(true)}
        />
        {showDebugMenuModal && (
          <DebugMenuModal
            onViewLevels={handleViewLevels}
            onViewUserData={handleViewUserData}
            onClose={() => setShowDebugMenuModal(false)}
          />
        )}
      </>
    );
  }

  if (!isDebugMode && gameFlowState === "comprehension") {
    return (
      <ComprehensionCheck
        onComplete={handleComprehensionComplete}
        onFailure={handleComprehensionFailure}
        onReturnToInstructions={handleReturnToInstructions}
        gameSessionId={gameSessionId}
      />
    );
  }

  if (gameFlowState === "tutorialComplete") {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100 p-4">
        <div className="max-w-md mx-auto p-6 bg-white rounded-lg shadow-md text-center text-black">
          <h2 className="text-2xl font-bold mb-4 text-black">
            Tutorial Completed!
          </h2>
          <p className="mb-4 text-black">
            You have finished the tutorial. Click the buttons below to retry the
            two levels or proceed to the main experiment.
          </p>
          <div className="flex justify-center space-x-4">
            <button
              onClick={() => {
                if (!isDebugMode) {
                  levelSequencer.retryTutorial();
                  setGameFlowState("game");
                }
              }}
              className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
            >
              Retry Tutorial
            </button>
            <button
              onClick={() => {
                if (!isDebugMode) {
                  levelSequencer.proceedToMainExperiment();
                  setGameFlowState("game");
                }
              }}
              className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded"
            >
              Proceed to Main Experiment
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (gameFlowState === "trialComplete") {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100 p-4">
        <div className="max-w-md mx-auto p-6 bg-white rounded-lg shadow-md text-center text-black">
          <h2 className="text-2xl font-bold mb-4 text-black">
            Trial Completed!
          </h2>
          <p className="mb-4 text-black">
            You earned {lastTrialDisplayedPoints} points this trial.
          </p>
          <p className="mb-4 text-black">
            Your total points: {totalDisplayedPoints}
          </p>
          <button
            onClick={() => setGameFlowState("game")}
            className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded"
          >
            Next Trial
          </button>
        </div>
      </div>
    );
  }

  if (gameFlowState === "userInfo") {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100 p-4">
        <div className="max-w-md mx-auto p-6 bg-white rounded-lg shadow-md text-center text-black">
          <h2 className="text-2xl font-bold mb-4 text-black">
            Experiment Complete!
          </h2>
          <p className="mb-4 text-black">
            Please provide your information to complete the study.
          </p>
          <p className="mb-4 text-black">
            Total points earned: {totalDisplayedPoints}
          </p>
        </div>
        <UserDataModal onComplete={() => setGameFlowState("thankYou")} />
      </div>
    );
  }

  if (gameFlowState === "thankYou") {
    return <Debrief totalPoints={totalDisplayedPoints} />;
  }

  return (
    <div className="min-h-screen min-w-[1024px] bg-white text-black">
      <style>{styles}</style>
      <div className="flex justify-between items-center px-8 py-8">
        <div>
          <h1 className="text-2xl font-bold">Multi-Agent Game</h1>
          {!isDebugMode && (
            <div className="text-lg text-gray-600 mt-1">
              {levelSequencer.isTutorial
                ? `Tutorial ${levelSequencer.currentLevelIndex + 1}/2`
                : `Trial ${levelSequencer.currentLevelIndex - 1}/${levelSequencer.currentSequence.length - 2}`}
            </div>
          )}
        </div>
        <button
          onClick={() => setShowDebugPasswordModal(true)}
          className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 transition-colors"
        >
          Debug Mode
        </button>
      </div>

      <div className="container mx-auto max-w-6xl px-4">
        {isDebugMode && (
          <div className="mb-4 text-center flex justify-center gap-4 items-center flex-wrap">
            <div className="flex items-center gap-2">
              <label className="font-semibold">Experiment:</label>
              <select
                value={selectedExperimentFolder}
                onChange={(e) => {
                  setSelectedExperimentFolder(e.target.value);
                  // Reset to first level in the selected folder
                  const filteredLevels = Object.keys(levels).filter(
                    levelId => getExperimentFolder(levelId) === e.target.value
                  );
                  if (filteredLevels.length > 0) {
                    handleDebugLevelChange(filteredLevels[0]);
                  }
                }}
                className="p-2 border rounded"
              >
                {getExperimentFolders().map((folder) => (
                  <option key={folder} value={folder}>
                    {folder}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-2">
              <label className="font-semibold">Scenario:</label>
              <select
                value={selectedScenario}
                onChange={(e) => handleScenarioChange(Number(e.target.value))}
                className="p-2 border rounded"
              >
                <option value={1}>Scenario 1</option>
                <option value={2}>Scenario 2</option>
                <option value={3}>Scenario 3</option>
              </select>
            </div>
            <div className="flex items-center gap-2">
              <label className="font-semibold">Level:</label>
              <select
                value={currentLevel}
                onChange={(e) => handleDebugLevelChange(e.target.value)}
                className="p-2 border rounded"
              >
                {Object.entries(levels)
                  .filter(([levelId]) => getExperimentFolder(levelId) === selectedExperimentFolder)
                  .map(([levelId, levelConfig]) => (
                    <option key={levelId} value={levelId}>
                      {levelConfig.id}
                    </option>
                  ))}
              </select>
            </div>
            <button
              onClick={handleReset}
              className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
            >
              Reset Level
            </button>
            <button
              onClick={() => {
                setIsDevMode(!isDevMode);
                // Reset the level when toggling dev mode
                const freshState = GameLogic.initializeGame(
                  loadLevel(currentLevel),
                  true,
                  !isDevMode,
                );
                setState(freshState);
                setMessage("");
                setGameWon(false);
              }}
              className={`px-4 py-2 ${isDevMode ? "bg-green-500 hover:bg-green-600" : "bg-blue-500 hover:bg-blue-600"} text-white rounded transition-colors`}
            >
              {isDevMode ? "Dev Mode: Count Up" : "Normal: Count Down"}
            </button>
          </div>
        )}

        {/* Message Box with Observation Timer */}
        {isObservationPeriod && !isDebugMode ? (
          <div className="w-full max-w-3xl mb-4 p-4 border border-gray-300 rounded h-20 flex items-center justify-center bg-white mx-auto">
            <div className="flex items-center gap-4">
              <div className="relative w-12 h-12">
                <div
                  className="absolute inset-0 border-4 border-blue-600 rounded-full"
                  style={{
                    animation: `shrink 10s linear forwards`,
                  }}
                ></div>
              </div>
              <div className="text-center">
                <h3 className="text-xl font-bold mb-1">
                  Observe the Map for 10 seconds
                </h3>
                <p className="text-sm text-gray-600">
                  Study the map layout and other player's position
                </p>
              </div>
            </div>
          </div>
        ) : (
          <StyledMessage message={message} />
        )}

        {/* Game Layout - Always Present */}
        <div className="flex justify-center">
          <div className="flex items-start gap-8">
            {/* Hide GameInfo completely during observation */}
            {!(isObservationPeriod && !isDebugMode) && (
              <div className="w-64">
                <GameInfo
                  goalType={state.goal.type}
                  points={state.stepsRemaining}
                  otherPlayers={memoizedAgentInfo}
                  isDevMode={isDevMode}
                />
                {isDebugMode && (
                  <div className="mt-4">
                    <DebugPanel
                      agents={state.agents}
                      onPathChange={handlePathChange}
                      gameState={state}
                      onLevelChange={handleDebugLevelChange}
                    />
                    {/* <StateLogger logs={stateLogs} /> */}
                  </div>
                )}
              </div>
            )}

            <div className="flex flex-col items-center">
              <GameGrid
                state={state}
                asciiMap={
                  levels[currentLevel]?.asciiMap ||
                  levels[currentLevel.replace(/_(1|2|ascii)$/, "")]?.asciiMap
                }
                activeInteraction={activeInteraction}
              />
            </div>

            {/* Hide action buttons and inventory completely during observation */}
            {!(isObservationPeriod && !isDebugMode) && (
              <div className="w-48 flex flex-col gap-4">
                <div className="w-full">
                  <ActionButtons
                    onMove={handleMove}
                    onObserve={handleObserve}
                    agents={state.agents}
                    disabled={actionsDisabled}
                    observeCooldownRemaining={observeCooldownRemaining}
                    moveCooldownRemaining={moveCooldownRemaining}
                  />
                </div>
                <div className="w-full">
                  <Inventory items={state.inventory || []} />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {showDebugPasswordModal && (
        <DebugPasswordModal
          onSubmit={handleDebugPassword}
          onClose={() => setShowDebugPasswordModal(false)}
        />
      )}

      {showDebugMenuModal && (
        <DebugMenuModal
          onViewLevels={handleViewLevels}
          onViewUserData={handleViewUserData}
          onClose={() => setShowDebugMenuModal(false)}
        />
      )}
    </div>
  );
}

export default MultiAgentGame;
