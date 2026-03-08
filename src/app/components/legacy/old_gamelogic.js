import { collection, addDoc, serverTimestamp } from 'firebase/firestore';
import { db } from './firebaseConfig';
import { addGameLog, updateGameSession } from './firestoreHelpers';

// const logToFirestore = async (action, data) => {
//   try {
//     await addDoc(collection(db, 'game_logs'), {
//       action,
//       data,
//       timestamp: serverTimestamp(),
//     });
//   } catch (error) {
//     console.error("Error logging to Firestore:", error);
//   }
// };

export const handlePlayerAction = (action, state, setState, setMessage, setActionLog, setStepsRemaining, setGameWon, npcMovesPerObservation, gameSessionId, currentLevel) => {
  let interactionInfo = null;
  setState(prev => {
    if (!prev) return null;
    let { player, npc, inventory, openedBarriers, barriers, treasurePots, wizards, blocks, goal } = prev;
    
    let validAction = false;
    let stepCost = 2; // Default step cost

    if (action === 'observe') {
      if (npc.currentMovementIndex >= npc.currentPath.length) {
        setMessage("The NPC has completed its movements. You can no longer observe.");
        return prev; // Return previous state without changes
      }
      validAction = true;
      stepCost = 1; // Observing costs 1 step
      npc = moveNPCMultiple(npc, blocks, barriers, treasurePots, wizards, npcMovesPerObservation);
      updateGameSession(gameSessionId, {
        npcPosition: { x: npc.x, y: npc.y, currentMovementIndex: npc.currentMovementIndex }
      });
      
      if (npc.currentMovementIndex >= npc.currentPath.length) {
        setMessage("The NPC has completed its movements. This was your last observation.");
      }
    } else {
      const [dx, dy] = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] }[action] || [0, 0];
      const newX = player.x + dx, newY = player.y + dy;

      const wizardIndex = wizards.findIndex(w => w.x === newX && w.y === newY);
      const barrier = barriers.find(b => b.x === newX && b.y === newY);
      const treasure = treasurePots.find(p => p.x === newX && p.y === newY);

      if (wizardIndex !== -1) {
        const wizard = wizards[wizardIndex];
        if (!wizard.interacted) {
          stepCost = 5;
          interactionInfo = handleInteraction(wizard, inventory, setMessage, setActionLog, gameSessionId);
          wizards[wizardIndex] = { ...wizard, interacted: true };
          validAction = true;
        } else {
          // setMessage("You've already interacted with this wizard.");
          validAction = false;
        }
      } else if (isValidMove(newX, newY, blocks, wizards)) {
        if (barrier) {
          if (canPassBarrier(barrier, inventory)) {
            player = { ...player, x: newX, y: newY };
            validAction = true;
          } else {
            const requiredItems = barrier.requiredItems.map(item => {
              const itemName = item.replace(/([A-Z])/g, ' $1').toLowerCase().trim();
              return itemName.startsWith('yellow') ? `an ${itemName}` : `a ${itemName}`;
            });
            
            let message = "You need ";
            if (requiredItems.length === 1) {
              message += requiredItems[0];
            } else if (requiredItems.length === 2) {
              message += `${requiredItems[0]} and ${requiredItems[1]}`;
            } else {
              const lastItem = requiredItems.pop();
              message += `${requiredItems.join(', ')}, and ${lastItem}`;
            }
            message += " to pass this barrier.";
            
            setMessage(message);
          }
        } else if (treasure) {
          interactionInfo = checkTreasureInteraction(treasure, goal, setMessage, setActionLog, setGameWon, gameSessionId);
          player = { ...player, x: newX, y: newY };
          validAction = true;
        } else {
          player = { ...player, x: newX, y: newY };
          validAction = true;
        }
      }
    }

    if (validAction) {
      const newStepsRemaining = prev.stepsRemaining - stepCost;
      setStepsRemaining(newStepsRemaining);

      const improvedLogEntry = {
        timestamp: new Date().toISOString(),
        type: "GAME_ACTION",
        data: {
          levelName: currentLevel,
          playerAction: action,
          playerPosition: player ? { x: player.x, y: player.y } : null,
          npcPosition: npc ? { x: npc.x, y: npc.y } : null,
          npcPathType: npc ? npc.selectedPathType : null,
          npcCurrentPathIndex: npc ? npc.currentMovementIndex : null,
          stepsRemaining: newStepsRemaining,
          stepCost: stepCost,
          interaction: interactionInfo
        }
      };

      // Filter out any undefined or null values
      improvedLogEntry.data = Object.fromEntries(
        Object.entries(improvedLogEntry.data).filter(([_, v]) => v != null)
      );

      addGameLog(gameSessionId, improvedLogEntry);

      if (interactionInfo) {
        addGameLog(gameSessionId, {
          type: interactionInfo.type,
          data: interactionInfo
        });
      }

      return { ...prev, player, npc, inventory, openedBarriers, stepsRemaining: newStepsRemaining };
    }

    return prev;

  });

  // return interactionInfo;
};

export const moveNPCMultiple = (npc, blocks, barriers, treasurePots, wizards, movesCount) => {
  let newNPC = { ...npc };
  for (let i = 0; i < movesCount; i++) {
    if (newNPC.currentMovementIndex >= newNPC.currentPath.length) {
      break;  // Stop if we've reached the end of the movement list
    }
    const currentMovement = newNPC.currentPath[newNPC.currentMovementIndex];
    const [dx, dy] = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] }[currentMovement];
    const newX = newNPC.x + dx, newY = newNPC.y + dy;

    if (isValidMove(newX, newY, blocks, wizards)) {
      newNPC = { ...newNPC, x: newX, y: newY, currentMovementIndex: newNPC.currentMovementIndex + 1 };
      
      // Check if the NPC has reached a treasure chest
      if (treasurePots.some(pot => pot.x === newX && pot.y === newY)) {
        // NPC has reached a treasure chest, mark it as disappeared
        newNPC.disappeared = true;
        break;  // Stop further movement
      }
    } else {
      newNPC = { ...newNPC, currentMovementIndex: newNPC.currentMovementIndex + 1 };
    }
  }
  return newNPC;
};

export const isValidMove = (x, y, blocks, wizards = []) => 
  x >= 0 && x < 11 && y >= 0 && y < 12 && 
  !blocks.some(b => b.x === x && b.y === y) && 
  !wizards.some(w => w.x === x && w.y === y);

export const canPassBarrier = (barrier, inventory) => {
  return barrier.requiredItems.every(item => inventory.includes(item));
};

export const handleInteraction = (wizard, inventory, setMessage, setActionLog, gameSessionId) => {
  const itemName = wizard.content.replace(/([A-Z])/g, ' $1').toLowerCase().trim();
  if (wizard.content && !inventory.includes(wizard.content)) {
    inventory.push(wizard.content);

    let message = '';
    let interactionType = null;
    
    if (wizard.content === 'redAmulet') {
    message = `You received a <span style="color: red; font-weight: bold;">red amulet</span> from the red wizard!`;
    inventory.push('redAmulet');
    interactionType = 'RED_AMULET';
  } else if (wizard.content === 'blueAmulet') {
    message = `You received a <span style="color: blue; font-weight: bold;">blue amulet</span> from the blue wizard!`;
    inventory.push('blueAmulet');
    interactionType = 'BLUE_AMULET';
  } else if (wizard.content === 'nothing') {
    message = `This wizard has <span style="color: black; font-weight: bold;">nothing</span> to give you.`;
    interactionType = 'NOTHING';
  }

  setMessage(message);
  setActionLog(prev => [...prev, { action: 'Interact with Wizard', item: wizard.content, timestamp: new Date().toISOString() }]);
    
    if (gameSessionId) { 
      addGameLog(gameSessionId, {
        type: 'WIZARD_INTERACTION',
        data: { wizardColor: wizard.color, itemReceived: itemName }
      });
    }

    return { type: 'WIZARD_INTERACTION', wizardColor: wizard.color, itemReceived: itemName };
  } else {
    setMessage("This wizard has nothing to give you.");
    return null;
  }
};

export const checkTreasureInteraction = (treasure, goal, setMessage, setActionLog, setGameWon, gameSessionId) => {
  const isCorrect = treasure.type === goal.type;
  if (isCorrect) {
    setMessage(`You found the ${treasure.type} treasure! You've completed the goal: ${goal.description}. You win!`);
    setActionLog(prev => [...prev, { action: 'Find Correct Treasure', type: treasure.type, timestamp: new Date().toISOString() }]);
    setGameWon(true);
  } else {
    setMessage(`You found treasure ${treasure.type}, but your goal is to find treasure ${goal.type}. Keep searching!`);
    setActionLog(prev => [...prev, { action: 'Find Wrong Treasure', type: treasure.type, goalType: goal.type, timestamp: new Date().toISOString() }]);
  }
  
  if (gameSessionId) {
    addGameLog(gameSessionId, {
      type: 'TREASURE_FOUND',
      data: { type: treasure.type, isCorrect, goalType: goal.type }
    });
  }

  return { type: 'TREASURE_FOUND', treasureType: treasure.type, isCorrect, goalType: goal.type };
};

export const initializeLevel = (levelData, gameSessionId, pathType) => {
  if (!levelData || !levelData.goal) {
    console.error('Invalid level data:', levelData);
    return null;
  }

  const selectedPathType = pathType;
  if (!levelData.npc.movements[selectedPathType]) {
    console.error(`Invalid path type: ${selectedPathType}. Available types:`, Object.keys(levelData.npc.movements));
    return null;
  }

  let adjustedStepLimit = levelData.stepLimit;
  if (selectedPathType === 'experienced1') {
    adjustedStepLimit = Math.ceil(adjustedStepLimit + 25);
  } else if (selectedPathType === 'novice1') {
    adjustedStepLimit = Math.ceil((adjustedStepLimit + 25) * 1.1 / 5) * 5;
  } else if (selectedPathType === 'experienced2' || selectedPathType === 'novice2') {
    adjustedStepLimit = Math.ceil((adjustedStepLimit + 25) * 1.2 / 5) * 5;
  }

  const initialState = {
    ...levelData,
    inventory: [],
    openedBarriers: [],
    stepsRemaining: adjustedStepLimit,
    message: `Goal: ${levelData.goal.description || 'No goal description available'}`,
    npc: {
      ...levelData.npc,
      selectedPathType,
      currentPath: levelData.npc.movements[selectedPathType].path,
      currentMovementIndex: 0,
      goal: levelData.npc.movements[selectedPathType].goal
    }
  };

  if (gameSessionId) {
    addGameLog(gameSessionId, {
      type: 'LEVEL_INIT',
      data: { 
        levelName: levelData.name,
        levelDetails: {
          player: levelData.player,
          npc: {
            startPosition: { x: levelData.npc.x, y: levelData.npc.y },
            pathType: selectedPathType,
            goal: initialState.npc.goal
          },
          barriers: levelData.barriers,
          treasurePots: levelData.treasurePots,
          wizards: levelData.wizards,
          goal: levelData.goal,
          adjustedStepLimit: adjustedStepLimit
        }
      }
    }).catch(error => console.error("Failed to add game log:", error));
  }

  return { initialState, adjustedStepLimit };
};
