import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Download, RefreshCw } from 'lucide-react';
import { PiTreasureChestBold } from "react-icons/pi";
import SelectComponent from 'react-select';
import mapData from './mapData';
import GameGrid from './GameGrid';
import ActionButtons from './ActionButtons';
import Inventory from './Inventory';
import Legend from './Legend';
import { handlePlayerAction, initializeLevel } from './gameLogic';
import { addGameLog, updateGameSession, updateSessionWithLevelInfo } from './firestoreHelpers';

const GridGame = ({ 
  onComplete, 
  currentLevel, 
  pathType, 
  debugMode, 
  gameSessionId, 
  trialNumber, 
  isTutorial,
  onDebugLevelChange,
  onDebugReset
}) => {
  const [state, setState] = useState(null);
  const [message, setMessage] = useState(null);
  const [actionLog, setActionLog] = useState([]);
  const [stepsRemaining, setStepsRemaining] = useState(0);
  const [gameWon, setGameWon] = useState(false);
  const [npcMovesPerObservation, setNpcMovesPerObservation] = useState(1);
  const [selectedLevel, setSelectedLevel] = useState(currentLevel);
  const [npcPathType, setNpcPathType] = useState('');
  const [npcGoal, setNpcGoal] = useState(null);
  const gameIdRef = useRef(null);
  const levels = Object.keys(mapData);
  const [isLevelCompleted, setIsLevelCompleted] = useState(false);
  const levelCompletedRef = useRef(false);

const StyledMessage = ({ message }) => {
    if (!message) return null;
  
    const parts = message.split(/(<span.*?<\/span>)/);
    return (
      <p>
        {parts.map((part, index) => {
          if (part.startsWith('<span')) {
            const color = part.match(/color: (\w+)/)?.[1];
            const isBold = part.includes('font-weight: bold');
            const isHighlighted = part.includes('background-color:');
            const content = part.replace(/<\/?span[^>]*>/g, '');
            return (
              <span 
                key={index}
                style={{
                  color: color || 'inherit',
                  fontWeight: isBold ? 'bold' : 'normal',
                  backgroundColor: isHighlighted ? 'rgba(255, 255, 0, 0.3)' : 'transparent', // Semi-transparent yellow
                  padding: isHighlighted ? '0 2px' : '0' // Add some padding for highlighted text
                }}
              >
                {content}
              </span>
            );
          }
          return part;
        })}
      </p>
    );
  };

  const initializeGameState = useCallback((level, path) => {
    console.log('Initializing game state for level:', level, 'with path type:', path);
    const { initialState, adjustedStepLimit } = initializeLevel(mapData[level], gameSessionId, path);
    if (initialState && initialState.npc && initialState.npc.movements[path]) {
      const updatedNPC = {
        ...initialState.npc,
        selectedPathType: path,
        currentPath: initialState.npc.movements[path].path,
        currentMovementIndex: 0
      };
      const newGoal = initialState.npc.movements[path].goal;
      
      setState({
        ...initialState,
        npc: updatedNPC
      });
      setStepsRemaining(adjustedStepLimit);
      setGameWon(false);
      setMessage(null);
      setActionLog([{ action: 'Level Loaded', level: level, npcPathType: path, timestamp: new Date().toISOString() }]);
      setIsLevelCompleted(false);
      levelCompletedRef.current = false;
    } else {
      // console.error('Failed to initialize game state', newState);
    }
  }, [gameSessionId]);
  
  useEffect(() => {
    if (gameSessionId && mapData[currentLevel]) {
      initializeGameState(currentLevel, pathType);
    }
  }, [currentLevel, gameSessionId, pathType, initializeGameState]);

  const handleGameLog = useCallback(async (action, data) => {
    if (gameSessionId) {
      await addGameLog(gameSessionId, {
        type: action,
        data: data
      });
    }
  }, [gameSessionId]);

  const handleNpcPathTypeChange = useCallback((e) => {
    const newPathType = e.target.value;
    console.log('Changing NPC path type to:', newPathType);
    initializeGameState(selectedLevel, newPathType);
  }, [selectedLevel, initializeGameState]);

  const [observesRemaining, setObservesRemaining] = useState(25);

  const handlePlayerActionWrapper = useCallback((action) => {
    // if (!isLevelCompleted && stepsRemaining > 0) {
    if (!isLevelCompleted) {
      if (action === 'observe' && observesRemaining <= 0) {
        setMessage("You have no observations remaining!");
        return;
      }
      const interactionInfo = handlePlayerAction(
        action,
        state,
        setState,
        setMessage,
        setActionLog,
        setStepsRemaining,
        setGameWon,
        npcMovesPerObservation,
        gameSessionId,
        currentLevel
      );
      
      // const improvedLogEntry = {
      //   timestamp: new Date().toISOString(),
      //   type: "GAME_ACTION",
      //   data: {
      //     levelName: currentLevel,
      //     playerAction: action,
      //     playerPosition: { x: state.player.x, y: state.player.y },
      //     npcPosition: { x: state.npc.x, y: state.npc.y },
      //     npcPathType: state.npc.selectedPathType,
      //     npcCurrentPathIndex: state.npc.currentMovementIndex,
      //     stepsRemaining: stepsRemaining,
      //     interaction: interactionInfo  
      //   }
      // };
  
      // if (interactionInfo) {
      //   addGameLog(gameSessionId, {
      //     type: interactionInfo.type,
      //     data: interactionInfo
      //   });
      // }
  
      // addGameLog(gameSessionId, improvedLogEntry);

      if (action === 'observe') {
        setObservesRemaining(prev => prev - 1);
      }
  
  
      if (interactionInfo && interactionInfo.type === 'TREASURE_FOUND' && interactionInfo.isCorrect) {
        setGameWon(true);
        setIsLevelCompleted(true);
      }
    }
  }, [isLevelCompleted, stepsRemaining, state, npcMovesPerObservation, gameSessionId, currentLevel]);

  useEffect(() => {
    // if ((gameWon || stepsRemaining <= 0) && !levelCompletedRef.current) {
    if ((gameWon) && !levelCompletedRef.current) {
      levelCompletedRef.current = true;
      setIsLevelCompleted(true);
      setTimeout(() => {
        onComplete(gameWon, stepsRemaining);
      }, 2500);
    }
  }, [gameWon, stepsRemaining, onComplete]);

  // useEffect(() => {
  //   if (stepsRemaining <= 0 && !gameWon && !levelCompletedRef.current) {
  //     levelCompletedRef.current = true;
  //     setMessage("You've run out of steps. Moving to the next level.");
  //     setTimeout(() => {
  //       onComplete(false);
  //     }, 2000);
  //   }
  // }, [stepsRemaining, gameWon, onComplete]);

  useEffect(() => {
    levelCompletedRef.current = false;
  }, [currentLevel]);

  const resetLevel = useCallback(() => {
    initializeGameState(selectedLevel, npcPathType);
  }, [selectedLevel, npcPathType, initializeGameState]);

  const exportActionLog = useCallback(() => {
    const jsonString = JSON.stringify(actionLog, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const href = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = href;
    link.download = 'game-action-log.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, [actionLog]);

  if (!state) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="text-xl font-semibold">Loading...</div>
      </div>
    );
  }

  const levelOptions = levels.map(level => ({ value: level, label: level }));

  const TreasureIcon = ({ letter }) => (
    <div className="relative inline-block w-8 h-8 ml-2">
      <PiTreasureChestBold className="w-8 h-8 text-yellow-500" />
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-black font-bold text-sm">{letter}</span>
      </div>
    </div>
  );
  
  const GameInfoPanel = ({ goalType, points, otherPlayerLevel }) => {
    return (
      <div className="bg-white p-4 rounded-lg shadow-md w-full">
        <h2 className="text-2xl font-bold mb-4 text-center">Game Info</h2>
        <div className="space-y-4">
          <div className="bg-gray-100 p-3 rounded-lg">
            <span className="text-sm font-semibold block mb-1">Your Goal:</span>
            <div className="flex items-center">
              <span className="text-2xl font-bold text-yellow-600">Treasure {goalType}</span>
              <TreasureIcon letter={goalType} />
            </div>
          </div>
          <div className="bg-gray-100 p-3 rounded-lg">
            <span className="text-sm font-semibold block mb-1">Your Points:</span>
            <span className="text-2xl font-bold text-green-600">{points}</span>
          </div>
          <div className="bg-gray-100 p-3 rounded-lg">
            <span className="text-sm font-semibold block mb-1">Other Player:</span>
            <span className="text-2xl font-bold text-blue-600">{otherPlayerLevel}</span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <>
      <div className="flex flex-col items-center justify-center min-h-screen p-4" style={{ transform: 'scale(0.8)', transformOrigin: 'center top' }}>
        <div className="text-xl font-bold mb-4">
          {isTutorial ? `Tutorial Trial ${trialNumber}` : `Trial ${trialNumber}`}
        </div>
        {debugMode && (
          <div className="flex justify-center items-center w-full max-w-5xl mb-4">
            <div className="flex space-x-4">
              <SelectComponent
                options={levelOptions}
                value={{ value: selectedLevel, label: selectedLevel }}
                onChange={(option) => {
                  setSelectedLevel(option.value);
                  initializeGameState(option.value, npcPathType);
                }}
                className="w-48"
              />
              <button
                onClick={resetLevel}
                className="px-4 py-2 bg-yellow-500 text-white rounded hover:bg-yellow-600 transition-colors"
              >
                <RefreshCw size={20} />
              </button>
              <button
                onClick={exportActionLog}
                className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 transition-colors"
              >
                <Download size={20} />
              </button>
              <select
                value={npcPathType}
                onChange={handleNpcPathTypeChange}
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
              >
                <option value="experienced1">Experienced Path 1</option>
                <option value="experienced2">Experienced Path 2</option>
                <option value="novice1">Novice Path 1</option>
                <option value="novice2">Novice Path 2</option>
              </select>
              <div key={npcGoal} className="px-4 py-2 bg-purple-500 text-white rounded">
                NPC Goal: {npcGoal}
              </div>
            </div>
          </div>
        )}

        <div className="w-full max-w-3xl mb-4 p-2 border border-gray-300 rounded h-16 flex items-center justify-center bg-white">
          <StyledMessage message={message} />
        </div>
        
        <div className="flex justify-center items-start space-x-8 w-full max-w-6xl">
          <div className="flex flex-col items-start w-1/5">
            <GameInfoPanel 
              goalType={state.goal.type}
              points={stepsRemaining}
              otherPlayerLevel={pathType.includes('experienced') ? 'Expert' : 'Advanced'}
            />
            <Legend />
          </div>

          <div className="flex flex-col items-center w-3/5">
            <GameGrid state={state} />
            <Inventory inventory={state.inventory} />
          </div>

          <div className="w-1/5">
            <ActionButtons  
              handlePlayerAction={handlePlayerActionWrapper} 
              disabled={gameWon}
              observesRemaining={observesRemaining}
            />
          </div>
        </div>

        {debugMode && (
          <div className="mt-4">
            <label htmlFor="npcMoves" className="mr-2">NPC Moves per Observation:</label>
            <input
              type="number"
              id="npcMoves"
              value={npcMovesPerObservation}
              onChange={(e) => setNpcMovesPerObservation(Math.max(1, parseInt(e.target.value)))}
              min="1"
              className="w-16 px-2 py-1 border rounded"
            />
          </div>
        )}
      </div>
    </>
  );
};

export default GridGame;
