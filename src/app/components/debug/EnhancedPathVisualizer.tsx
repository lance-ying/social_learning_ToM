"use client"
import React, { useState, useEffect, useRef } from 'react';
import { Download } from 'lucide-react';
import html2canvas from 'html2canvas';
import { PiTreasureChestBold } from "react-icons/pi";
import { GiNestedHexagons } from "react-icons/gi";

interface EnhancedPathVisualizerProps {
  gameState: any;
  onPathChange?: (agentId: number, pathType: string) => void;
  onLevelChange?: (levelId: string) => void;
  initialSelectedPath?: string;
  initialMovementIndex?: number;
}

const EnhancedPathVisualizer: React.FC<EnhancedPathVisualizerProps> = ({
  gameState,
  onPathChange,
  onLevelChange,
  initialSelectedPath = '',
  initialMovementIndex = 100
}) => {
  const [selectedAgent, setSelectedAgent] = useState<number>(1);
  const [selectedPath, setSelectedPath] = useState<string>(initialSelectedPath);
  const [movementIndex, setMovementIndex] = useState<number>(initialMovementIndex);
  const [trailSegments, setTrailSegments] = useState<any[]>([]);
  const [allAgentTrails, setAllAgentTrails] = useState<{[agentId: number]: any[]}>({});
  const [isGeneratingImage, setIsGeneratingImage] = useState<boolean>(false);
  
  const visualizationRef = useRef<HTMLDivElement>(null);
  
  const CELL_SIZE = 48;
  const GRID_WIDTH = 11;
  const GRID_HEIGHT = 12;
  const MAX_ARROW_SIZE = 17;
  const MAX_OPACITY = 0.9;
  const MIN_OPACITY = 0.2;
  const TRAIL_LENGTH = 100;

  // Initialize selected path when agent changes
  useEffect(() => {
    if (gameState?.agents && gameState.agents.length > 0) {
      const agent = gameState.agents.find((a: any) => a.id === selectedAgent);
      if (agent && agent.selectedPath) {
        setSelectedPath(agent.selectedPath);
      }
    }
  }, [selectedAgent, gameState?.agents]);

  // Generate trail segments for ALL agents when path or movement index changes
  useEffect(() => {
    if (!gameState?.agents || !selectedPath) return;

    const generateSegmentsForAgent = (agent: any) => {
      if (!agent || !agent.movements || !agent.movements[selectedPath]?.path) return [];

      // Use agent's current position as starting point
      const path = agent.movements[selectedPath].path;
      let segments = [];
      let currentX = agent.x;
      let currentY = agent.y;
      let directionCounts = new Map();

      segments.push({
        x: currentX,
        y: currentY,
        move: null,
        index: -1,
        visitCount: 1,
        agentId: agent.id,
        agentColor: agent.color
      });

      for (let i = 0; i < Math.min(movementIndex, path.length); i++) {
        const move = path[i];
        const [dx, dy] = getMoveOffset(move);
        const nextX = currentX + dx;
        const nextY = currentY + dy;

        // Check if next position is a wizard
        const isWizard = gameState.wizards.some((w: any) => w.x === nextX && w.y === nextY);

        // If it's a wizard, don't update position but mark as interacting
        if (isWizard) {
          segments.push({
            x: currentX, // Keep current position
            y: currentY, // Keep current position
            move: move,
            index: i,
            visitCount: 1,
            isInteracting: true,
            interactingAt: { x: nextX, y: nextY },
            agentId: agent.id,
            agentColor: agent.color
          });
        } else {
          // Normal movement
          currentX = nextX;
          currentY = nextY;

          const cellKey = `${currentX},${currentY}`;
          const currentCount = directionCounts.get(cellKey) || 0;
          directionCounts.set(cellKey, currentCount + 1);

          segments.push({
            x: currentX,
            y: currentY,
            move: move,
            index: i,
            visitCount: currentCount + 1,
            isInteracting: false,
            agentId: agent.id,
            agentColor: agent.color
          });
        }
      }

      // Use all segments when movement index is high (for downloads), otherwise limit by TRAIL_LENGTH
      const segmentsToShow = movementIndex > 200 ? segments : segments.slice(-TRAIL_LENGTH);

      return segmentsToShow.map((seg, idx, arr) => ({
        ...seg,
        distance: arr.length - idx - 1
      }));
    };

    // Generate trails for all agents
    const allTrails: {[agentId: number]: any[]} = {};
    gameState.agents.forEach((agent: any) => {
      allTrails[agent.id] = generateSegmentsForAgent(agent);
    });
    setAllAgentTrails(allTrails);

    // Keep the selected agent's trail for backward compatibility
    const selectedAgentData = gameState.agents.find((a: any) => a.id === selectedAgent);
    if (selectedAgentData) {
      setTrailSegments(allTrails[selectedAgent] || []);
    }
  }, [gameState, selectedPath, movementIndex, selectedAgent]);

  const getMoveOffset = (move: string) => {
    switch (move) {
      case 'up': return [0, -1];
      case 'down': return [0, 1];
      case 'left': return [-1, 0];
      case 'right': return [1, 0];
      default: return [0, 0];
    }
  };

  const downloadVisualization = async () => {
    if (!visualizationRef.current) return;

    try {
      setIsGeneratingImage(true);
      const canvas = await html2canvas(visualizationRef.current, {
        logging: false,
      });

      const agent = gameState.agents.find((a: any) => a.id === selectedAgent);
      const agentType = agent?.type || 'unknown';
      const filename = `agent${selectedAgent}_${selectedPath}_${agentType}_${movementIndex}.png`;

      const link = document.createElement('a');
      link.download = filename;
      link.href = canvas.toDataURL('image/png');
      link.click();
    } catch (error) {
      console.error('Error downloading visualization:', error);
    } finally {
      setIsGeneratingImage(false);
    }
  };

  const pullAllPathings = async () => {
    if (!visualizationRef.current) return;

    setIsGeneratingImage(true);

    const pathTypes = ['experienced1', 'experienced2'];
    const levelId = gameState?.levelId || gameState?.goal?.description?.match(/Treasure\s+([ABC])/)?.[1] || 'level';

    // Set movement index to max to show full path
    const agent = gameState.agents.find((a: any) => a.id === selectedAgent);
    if (!agent) {
      setIsGeneratingImage(false);
      return;
    }

    for (const pathType of pathTypes) {
      // Check if this path exists
      if (!agent.movements?.[pathType]) {
        console.warn(`Path ${pathType} not found for agent ${selectedAgent}`);
        continue;
      }

      // Switch to the path
      setSelectedPath(pathType);
      const maxLength = agent.movements[pathType]?.path?.length || 100;
      setMovementIndex(maxLength);

      // Wait for React to update and render
      await new Promise(resolve => setTimeout(resolve, 1000));

      try {
        const canvas = await html2canvas(visualizationRef.current, {
          logging: false,
        });

        const filename = `${levelId}_${pathType}.png`;

        const link = document.createElement('a');
        link.download = filename;
        link.href = canvas.toDataURL('image/png');
        link.click();

        // Wait between downloads
        await new Promise(resolve => setTimeout(resolve, 500));
      } catch (error) {
        console.error(`Error downloading ${pathType}:`, error);
      }
    }

    setIsGeneratingImage(false);
  };

  const downloadAllNewLevels = async () => {
    if (!visualizationRef.current || !onLevelChange) {
      console.warn('Level change handler not available');
      return;
    }

    setIsGeneratingImage(true);

    // Exp2 level IDs (single agent)
    const exp2LevelIds = ['s111', 's112'];

    // Exp3 level IDs (two agents)
    const exp3LevelIds = [
      's211', 's221', 's311', 's321', 's331', 's332',
      's341', 's342', 's351', 's361', 's371', 's411', 's421', 's431',
      's432', 's441', 's442', 's511', 's521', 's531', 's532', 's541',
      's542', 's543', 's544'
    ];

    const pathTypes = ['experienced1', 'experienced2'];
    let successCount = 0;
    let totalExpected = (exp2LevelIds.length * pathTypes.length) + (exp3LevelIds.length * pathTypes.length);

    // Process exp2 levels (single agent)
    for (const levelId of exp2LevelIds) {
      console.log(`\n========================================`);
      console.log(`Processing EXP2 level: ${levelId}`);
      console.log(`========================================`);

      onLevelChange(levelId);
      await new Promise(resolve => setTimeout(resolve, 2500));

      for (const pathType of pathTypes) {
        console.log(`  → Switching to path: ${pathType}`);

        if (onPathChange) {
          onPathChange(1, pathType);
        }

        setSelectedPath(pathType);
        setMovementIndex(1000); // High number to show full paths
        await new Promise(resolve => setTimeout(resolve, 2500));

        try {
          const canvas = await html2canvas(visualizationRef.current, {
            logging: false,
          });

          const filename = `stimuli_${levelId}_agent1_${pathType}_and_agent2_${pathType}.png`;
          const link = document.createElement('a');
          link.download = filename;
          link.href = canvas.toDataURL('image/png');
          link.click();

          successCount++;
          console.log(`  ✓ Downloaded ${successCount}/${totalExpected}: ${filename}`);
          await new Promise(resolve => setTimeout(resolve, 500));
        } catch (error) {
          console.error(`  ✗ Error downloading ${levelId} ${pathType}:`, error);
        }
      }

      await new Promise(resolve => setTimeout(resolve, 500));
    }

    // Process exp3 levels (two agents)
    for (const levelId of exp3LevelIds) {
      console.log(`\n========================================`);
      console.log(`Processing EXP3 level: ${levelId} (TWO AGENTS)`);
      console.log(`========================================`);

      onLevelChange(levelId);
      await new Promise(resolve => setTimeout(resolve, 2500));

      for (const pathType of pathTypes) {
        console.log(`  → Setting both agents to: ${pathType}`);

        // Set both agent 1 and agent 2 to the same pathType
        if (onPathChange) {
          onPathChange(1, pathType);
          await new Promise(resolve => setTimeout(resolve, 100));
          onPathChange(2, pathType);
        }

        setSelectedPath(pathType);
        setMovementIndex(1000); // High number to show full paths
        await new Promise(resolve => setTimeout(resolve, 2500));

        try {
          const canvas = await html2canvas(visualizationRef.current, {
            logging: false,
          });

          const filename = `stimuli_${levelId}_agent1_${pathType}_and_agent2_${pathType}.png`;
          const link = document.createElement('a');
          link.download = filename;
          link.href = canvas.toDataURL('image/png');
          link.click();

          successCount++;
          console.log(`  ✓ Downloaded ${successCount}/${totalExpected}: ${filename}`);
          await new Promise(resolve => setTimeout(resolve, 500));
        } catch (error) {
          console.error(`  ✗ Error downloading ${levelId} ${pathType}:`, error);
        }
      }

      await new Promise(resolve => setTimeout(resolve, 500));
    }

    setIsGeneratingImage(false);
    console.log(`\n========================================`);
    console.log(`✓ COMPLETED! Downloaded ${successCount}/${totalExpected} images.`);
    console.log(`========================================\n`);
  };

  const downloadAllExp3Levels = async () => {
    if (!visualizationRef.current || !onLevelChange) {
      console.warn('Level change handler not available');
      return;
    }

    setIsGeneratingImage(true);

    // Exp3_true level IDs (three agents: M, X, Y)
    const exp3TrueLevelIds = [
      'sm211_true', 'sm221_true', 'sm311_true', 'sm321_true', 'sm331_true', 'sm332_true',
      'sm341_true', 'sm342_true', 'sm351_true', 'sm361_true', 'sm371_true',
      'sm411_true', 'sm421_true', 'sm431_true', 'sm432_true',
      'sm511_true', 'sm521_true', 'sm531_true', 'sm541_true', 'sm543_true'
    ];

    const pathTypes = ['experienced1', 'experienced2'];
    let successCount = 0;
    let totalExpected = exp3TrueLevelIds.length * pathTypes.length;

    console.log(`\n========================================`);
    console.log(`Starting EXP3_TRUE Download: ${exp3TrueLevelIds.length} levels x 2 paths = ${totalExpected} images`);
    console.log(`========================================\n`);

    // Process exp3_true levels (three agents)
    for (const levelId of exp3TrueLevelIds) {
      console.log(`\n========================================`);
      console.log(`Processing EXP3_TRUE level: ${levelId} (THREE AGENTS)`);
      console.log(`========================================`);

      onLevelChange(levelId);
      await new Promise(resolve => setTimeout(resolve, 2500));

      for (const pathType of pathTypes) {
        console.log(`  → Setting all agents to: ${pathType}`);

        // Set agent 2 (X) and agent 3 (Y) to the same pathType
        if (onPathChange) {
          onPathChange(2, pathType);
          await new Promise(resolve => setTimeout(resolve, 100));
          onPathChange(3, pathType);
        }

        setSelectedPath(pathType);
        setMovementIndex(1000); // High number to show full paths
        await new Promise(resolve => setTimeout(resolve, 2500));

        try {
          const canvas = await html2canvas(visualizationRef.current, {
            logging: false,
          });

          const filename = `stimuli_${levelId}_agent2_${pathType}_and_agent3_${pathType}.png`;
          const link = document.createElement('a');
          link.download = filename;
          link.href = canvas.toDataURL('image/png');
          link.click();

          successCount++;
          console.log(`  ✓ Downloaded ${successCount}/${totalExpected}: ${filename}`);
          await new Promise(resolve => setTimeout(resolve, 500));
        } catch (error) {
          console.error(`  ✗ Error downloading ${levelId} ${pathType}:`, error);
        }
      }

      await new Promise(resolve => setTimeout(resolve, 500));
    }

    setIsGeneratingImage(false);
    console.log(`\n========================================`);
    console.log(`✓ EXP3_TRUE COMPLETED! Downloaded ${successCount}/${totalExpected} images.`);
    console.log(`========================================\n`);
  };

  const downloadAllExp4Levels = async () => {
    if (!visualizationRef.current || !onLevelChange) {
      console.warn('Level change handler not available');
      return;
    }

    setIsGeneratingImage(true);

    // Exp4 level IDs (two agents: X, Y)
    const exp4LevelIds = [
      'sm211_exp4', 'sm221_exp4', 'sm311_exp4', 'sm321_exp4', 'sm331_exp4',
      'sm341_exp4', 'sm351_exp4', 'sm361_exp4', 'sm371_exp4', 'sm411_exp4',
      'sm421_exp4', 'sm431_exp4', 'sm432_exp4', 'sm511_exp4', 'sm521_exp4',
      'sm531_exp4', 'sm541_exp4', 'sm543_exp4', 'sm611_exp4', 'sm612_exp4'
    ];

    // All path types for exp4
    const pathTypes = ['experienced1', 'experienced2'];

    let successCount = 0;
    let totalExpected = exp4LevelIds.length * pathTypes.length; // 20 × 2 = 40

    console.log(`\n========================================`);
    console.log(`Starting EXP4 Download: ${exp4LevelIds.length} levels × 2 paths = ${totalExpected} images`);
    console.log(`========================================\n`);

    // Process exp4 levels (two agents)
    for (const levelId of exp4LevelIds) {
      console.log(`\n========================================`);
      console.log(`Processing EXP4 level: ${levelId} (TWO AGENTS)`);
      console.log(`========================================`);

      onLevelChange(levelId);
      await new Promise(resolve => setTimeout(resolve, 2500));

      for (const pathType of pathTypes) {
        console.log(`  → Setting both agents to: ${pathType}`);

        // Set both agent 1 (X) and agent 2 (Y) to the same pathType
        if (onPathChange) {
          onPathChange(1, pathType);
          await new Promise(resolve => setTimeout(resolve, 100));
          onPathChange(2, pathType);
        }

        setSelectedPath(pathType);
        setMovementIndex(1000); // High number to show full paths
        await new Promise(resolve => setTimeout(resolve, 2500));

        try {
          const canvas = await html2canvas(visualizationRef.current, {
            logging: false,
          });

          const filename = `stimuli_${levelId}_agent1_${pathType}_and_agent2_${pathType}.png`;
          const link = document.createElement('a');
          link.download = filename;
          link.href = canvas.toDataURL('image/png');
          link.click();

          successCount++;
          console.log(`  ✓ Downloaded ${successCount}/${totalExpected}: ${filename}`);
          await new Promise(resolve => setTimeout(resolve, 500));
        } catch (error) {
          console.error(`  ✗ Error downloading ${levelId} ${pathType}:`, error);
        }
      }

      await new Promise(resolve => setTimeout(resolve, 500));
    }

    setIsGeneratingImage(false);
    console.log(`\n========================================`);
    console.log(`✓ EXP4 COMPLETED! Downloaded ${successCount}/${totalExpected} images.`);
    console.log(`========================================\n`);
  };

  const renderArrow = (segment: any, prevSegment: any, opacity: number, scale: number) => {
    const direction = segment.move;
    if (!direction) return null;

    const startX = prevSegment.x;
    const startY = prevSegment.y;
    const endX = segment.isInteracting && segment.interactingAt
      ? segment.interactingAt.x
      : segment.x;
    const endY = segment.isInteracting && segment.interactingAt
      ? segment.interactingAt.y
      : segment.y;

    const rotation = {
      'up': 0,
      'down': 180,
      'left': 270,
      'right': 90
    }[direction as 'up' | 'down' | 'left' | 'right'];

    // Calculate offset based on agent ID to prevent overlap
    const agentId = segment.agentId;
    const offsetMultiplier = (agentId === 2) ? -0.25 : (agentId === 3) ? 0.25 : 0;
    
    // Offset perpendicular to the direction of movement
    let offsetX = 0;
    let offsetY = 0;
    if (direction === 'up' || direction === 'down') {
      offsetX = CELL_SIZE * offsetMultiplier; // Horizontal offset for vertical movement
    } else {
      offsetY = CELL_SIZE * offsetMultiplier; // Vertical offset for horizontal movement
    }

    const agentColor = segment.agentColor || 'blue';
    const arrowColor = agentColor === 'blue' ? 'rgba(59, 130, 246, ' + opacity + ')' : 'rgba(34, 197, 94, ' + opacity + ')';
    const interactionColor = agentColor === 'blue' ? 'rgba(59, 130, 246, 0.7)' : 'rgba(34, 197, 94, 0.7)';

    return (
      <div
        className="absolute flex items-center justify-center"
        style={{
          left: `${startX * CELL_SIZE + CELL_SIZE/2 + offsetX}px`,
          top: `${startY * CELL_SIZE + CELL_SIZE/2 + offsetY}px`,
          width: CELL_SIZE,
          height: CELL_SIZE,
          opacity,
          transform: `translate(-50%, -50%) scale(${scale})`,
          zIndex: 20 + agentId // Different z-index per agent
        }}
      >
        <div
          className="w-0 h-0"
          style={{
            borderLeft: `${MAX_ARROW_SIZE/2 * scale}px solid transparent`,
            borderRight: `${MAX_ARROW_SIZE/2 * scale}px solid transparent`,
            borderBottom: `${MAX_ARROW_SIZE * scale}px solid ${arrowColor}`,
            transform: `rotate(${rotation}deg)`,
          }}
        />

        {segment.isInteracting && (
          <div
            className="absolute"
            style={{
              left: `${(endX - startX) * CELL_SIZE/2}px`,
              top: `${(endY - startY) * CELL_SIZE/2}px`,
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              backgroundColor: interactionColor,
              border: '2px solid white',
              zIndex: 30
            }}
          />
        )}
      </div>
    );
  };

  const renderCell = (x: number, y: number) => {
    const cellContent = [];

    // Check if it's a block first
    if (gameState.blocks.some((b: any) => b.x === x && b.y === y)) {
      cellContent.push(
        <div key="block" className="absolute inset-0 bg-gray-800" />
      );
      return cellContent;
    }

    // Add barriers
    const barrier = gameState.barriers.find((b: any) => b.x === x && b.y === y);
    if (barrier) {
      cellContent.push(
        <div key="barrier" className="absolute inset-0 flex items-center justify-center">
          <GiNestedHexagons
            className={barrier.requiredItems.includes('blueAmulet') ? 'text-blue-500' : 'text-red-500'}
            style={{ fontSize: `${CELL_SIZE * 1}px` }}
          />
        </div>
      );
    }

    // Add treasures
    const treasure = gameState.treasurePots.find((t: any) => t.x === x && t.y === y);
    if (treasure) {
      const LETTER_OFFSET_Y = -3; // Adjustable pixel spacing for letter - negative moves up, positive moves down
      const isGoalTreasure = gameState.goal && treasure.label === gameState.goal.type;
      
      cellContent.push(
        <div key="treasure" className="absolute inset-0 flex items-center justify-center">
          <div className="relative">
            <PiTreasureChestBold
              className="text-yellow-500"
              style={{ fontSize: `${CELL_SIZE * 1}px` }}
            />
            <div
              className="absolute inset-0 flex items-center justify-center text-lg font-bold text-black"
              style={{ top: `${LETTER_OFFSET_Y}px` }}
            >
              {treasure.label}
            </div>
            {isGoalTreasure && (
              <div
                className="absolute rounded-full border-8 border-red-500"
                style={{
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  width: `${CELL_SIZE * 1.0}px`,
                  height: `${CELL_SIZE * 1.0}px`,
                  pointerEvents: 'none'
                }}
              />
            )}
          </div>
        </div>
      );
    }

    // Add wizards
    const wizard = gameState.wizards.find((w: any) => w.x === x && w.y === y);
    if (wizard) {
      cellContent.push(
        <div key="wizard" className="absolute inset-0 flex items-center justify-center">
          <img
            src={wizard.color.includes('red') ? '/icons/f.png' : '/icons/d.png'}
            alt={wizard.color.includes('red') ? 'Sorcerer 1' : 'Sorcerer 2'}
            style={{
              width: `${CELL_SIZE * 1}px`,
              height: `${CELL_SIZE * 1}px`
            }}
          />
        </div>
      );
    }

    return cellContent;
  };

  const handlePathChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newPath = e.target.value;
    setSelectedPath(newPath);
    if (onPathChange) {
      onPathChange(selectedAgent, newPath);
    }
  };

  const currentPosition = trailSegments[trailSegments.length - 1];
  const agent = gameState?.agents?.find((a: any) => a.id === selectedAgent);
  const pathOptions = agent?.movements ? Object.keys(agent.movements) : [];
  const maxPathLength = agent?.movements?.[selectedPath]?.path?.length || 0;

  return (
    <div className="flex flex-col items-center justify-center p-4 bg-gray-100 rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4">Agent Path Visualizer</h2>

      {/* Debug info */}
      <div className="mb-2 text-xs text-gray-600 space-y-1">
        <div>Agent: {agent?.id} ({agent?.color}) | Start Pos: {agent ? `(${agent.x}, ${agent.y})` : 'Not found'}</div>
        <div>Current Pos: {currentPosition ? `(${currentPosition.x}, ${currentPosition.y})` : 'Not found'} | Trail Segments: {trailSegments.length}</div>
        <div>Selected Path: {selectedPath} | Path Length: {maxPathLength}</div>
      </div>
      
      <div className="mb-4 space-x-4 flex flex-wrap gap-2">
        <select 
          value={selectedAgent}
          onChange={(e) => setSelectedAgent(Number(e.target.value))}
          className="p-2 border rounded"
        >
          {gameState?.agents?.map((agent: any) => (
            <option key={agent.id} value={agent.id}>
              Agent {agent.id} ({agent.type || 'Unknown'})
            </option>
          ))}
        </select>
        
        <select 
          value={selectedPath}
          onChange={handlePathChange}
          className="p-2 border rounded"
        >
          {pathOptions.map((path: string) => (
            <option key={path} value={path}>{path}</option>
          ))}
        </select>

        <div className="inline-flex items-center">
          <label className="mr-2">Movement Index:</label>
          <input
            type="range"
            min="0"
            max={maxPathLength}
            value={movementIndex}
            onChange={(e) => setMovementIndex(parseInt(e.target.value))}
            className="w-48"
          />
          <span className="ml-2">{movementIndex} / {maxPathLength}</span>
        </div>

        <button
          onClick={downloadVisualization}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 flex items-center space-x-2"
          disabled={isGeneratingImage}
        >
          <Download size={16} />
          <span>{isGeneratingImage ? 'Generating...' : 'Download Current'}</span>
        </button>

        <button
          onClick={pullAllPathings}
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 flex items-center space-x-2"
          disabled={isGeneratingImage}
        >
          <Download size={16} />
          <span>{isGeneratingImage ? 'Generating...' : 'Pull All Pathings'}</span>
        </button>

        <button
          onClick={downloadAllNewLevels}
          className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 flex items-center space-x-2"
          disabled={isGeneratingImage || !onLevelChange}
        >
          <Download size={16} />
          <span>{isGeneratingImage ? 'Generating...' : 'Download All Levels (Exp2+Exp3) (54 images)'}</span>
        </button>

        <button
          onClick={downloadAllExp3Levels}
          className="px-4 py-2 bg-orange-500 text-white rounded hover:bg-orange-600 flex items-center space-x-2"
          disabled={isGeneratingImage || !onLevelChange}
        >
          <Download size={16} />
          <span>{isGeneratingImage ? 'Generating...' : 'Download Exp3_True (3 Agents) (40 images)'}</span>
        </button>

        <button
          onClick={downloadAllExp4Levels}
          className="px-4 py-2 bg-pink-500 text-white rounded hover:bg-pink-600 flex items-center space-x-2"
          disabled={isGeneratingImage || !onLevelChange}
        >
          <Download size={16} />
          <span>{isGeneratingImage ? 'Generating...' : 'Download Exp4 (2 Agents) (40 images)'}</span>
        </button>
      </div>
      
      <div 
        ref={visualizationRef}
        data-testid="enhanced-path-visualization"
        className="bg-black p-1 rounded-lg"
      >
        <div className="relative bg-white" style={{
          width: GRID_WIDTH * CELL_SIZE,
          height: GRID_HEIGHT * CELL_SIZE,
        }}>
          {/* Grid cells */}
          {Array.from({length: GRID_HEIGHT}).map((_, y) => 
            Array.from({length: GRID_WIDTH}).map((_, x) => (
              <div
                key={`${x}-${y}`}
                className="absolute border border-gray-200"
                style={{
                  left: x * CELL_SIZE,
                  top: y * CELL_SIZE,
                  width: CELL_SIZE,
                  height: CELL_SIZE,
                  borderColor: gameState.blocks.some((b: any) => b.x === x && b.y === y) ? 
                    'rgb(55, 65, 81)' : '#e5e7eb'
                }}
              >
                {renderCell(x, y)}
              </div>
            ))
          )}

          {/* Initial position markers for ALL agents */}
          {gameState?.agents?.map((agentData: any) => (
            <div
              key={`initial-pos-${agentData.id}`}
              className="absolute flex items-center justify-center"
              style={{
                left: `${agentData.x * CELL_SIZE}px`,
                top: `${agentData.y * CELL_SIZE}px`,
                width: `${CELL_SIZE}px`,
                height: `${CELL_SIZE}px`,
                zIndex: 25,
              }}
            >
              {/* Circle outline for initial position */}
              <div
                className="absolute rounded-full border-2"
                style={{
                  width: `${CELL_SIZE * 0.8}px`,
                  height: `${CELL_SIZE * 0.8}px`,
                  borderColor: agentData.color === 'blue' ? 'rgba(59, 130, 246, 0.5)' : 'rgba(34, 197, 94, 0.5)',
                  borderStyle: 'dashed'
                }}
              />
              <img
                src={agentData.color === 'blue' ? '/icons/al.png' : '/icons/green_a.png'}
                alt={`Player ${agentData.id}`}
                style={{
                  width: `${CELL_SIZE * 0.6}px`,
                  height: `${CELL_SIZE * 0.7}px`,
                  opacity: 0.3
                }}
              />
            </div>
          ))}

          {/* Player position */}
          {gameState.player && (
            <div
              className="absolute flex items-center justify-center"
              style={{
                left: `${gameState.player.x * CELL_SIZE}px`,
                top: `${gameState.player.y * CELL_SIZE}px`,
                width: `${CELL_SIZE}px`,
                height: `${CELL_SIZE}px`,
                zIndex: 40
              }}
            >
              <img
                src="/icons/bl.png"
                alt="Player"
                style={{
                  width: `${CELL_SIZE * 0.8}px`,
                  height: `${CELL_SIZE * 0.9}px`
                }}
              />
            </div>
          )}
          
          {/* Trail visualization for ALL agents */}
          {Object.entries(allAgentTrails).map(([agentId, segments]: [string, any[]]) => {
            return segments.map((segment, index) => {
              if (index === 0) return null; // Skip only initial position (has no move)
              const prevSegment = segments[index - 1];
              const opacity = MAX_OPACITY -
                ((segment.distance / TRAIL_LENGTH) * (MAX_OPACITY - MIN_OPACITY));
              const scale = Math.max(0.7, 1 - (segment.distance / (TRAIL_LENGTH * 2)));

              return (
                <div key={`trail-agent${agentId}-${segment.index}`}>
                  {renderArrow(segment, prevSegment, opacity, scale)}
                </div>
              );
            });
          })}

          {/* Current position for ALL agents */}
          {Object.entries(allAgentTrails).map(([agentId, segments]: [string, any[]]) => {
            const agentData = gameState.agents.find((a: any) => a.id === parseInt(agentId));
            const currentPos = segments[segments.length - 1];

            if (!currentPos || !agentData) return null;

            return (
              <div
                key={`agent-pos-${agentId}`}
                className="absolute flex items-center justify-center transition-all duration-200"
                style={{
                  left: `${currentPos.x * CELL_SIZE}px`,
                  top: `${currentPos.y * CELL_SIZE}px`,
                  width: `${CELL_SIZE}px`,
                  height: `${CELL_SIZE}px`,
                  zIndex: 50
                }}
              >
                {/* Glow effect behind agent for visibility */}
                <div
                  className="absolute rounded-full"
                  style={{
                    width: `${CELL_SIZE * 0.8}px`,
                    height: `${CELL_SIZE * 0.8}px`,
                    backgroundColor: agentData.color === 'blue' ? 'rgba(59, 130, 246, 0.3)' : 'rgba(34, 197, 94, 0.3)',
                    filter: 'blur(4px)',
                    zIndex: -1
                  }}
                />

                <img
                  src={agentData.color === 'blue' ? '/icons/al.png' : '/icons/green_a.png'}
                  alt={agentData.color === 'blue' ? 'Player 1' : 'Player 2'}
                  style={{
                    width: `${CELL_SIZE * 0.6}px`,
                    height: `${CELL_SIZE * 0.7}px`,
                    filter: 'drop-shadow(0 0 2px rgba(0, 0, 0, 0.5))'
                  }}
                  className={currentPos.isInteracting ? 'animate-pulse' : ''}
                />

                {currentPos.isInteracting && (
                  <div className="absolute -top-1 -right-1">
                    <div className="bg-yellow-400 text-xs px-1 rounded-full">
                      🔮
                    </div>
                  </div>
                )}
              </div>
            );
          })}

          {/* Keep original single agent position for backward compatibility when needed */}
          {currentPosition && agent && false && (
            <div
              className="absolute flex items-center justify-center transition-all duration-200"
              style={{
                left: `${currentPosition.x * CELL_SIZE}px`,
                top: `${currentPosition.y * CELL_SIZE}px`,
                width: `${CELL_SIZE}px`,
                height: `${CELL_SIZE}px`,
                zIndex: 50
              }}
            >
              {/* Glow effect behind agent for visibility */}
              <div
                className="absolute rounded-full"
                style={{
                  width: `${CELL_SIZE * 0.8}px`,
                  height: `${CELL_SIZE * 0.8}px`,
                  backgroundColor: agent.color === 'blue' ? 'rgba(59, 130, 246, 0.3)' : 'rgba(34, 197, 94, 0.3)',
                  filter: 'blur(4px)',
                  zIndex: -1
                }}
              />

              <img
                src={agent.color === 'blue' ? '/icons/al.png' : '/icons/green_a.png'}
                alt={agent.color === 'blue' ? 'Player 1' : 'Player 2'}
                style={{
                  width: `${CELL_SIZE * 0.6}px`,
                  height: `${CELL_SIZE * 0.7}px`,
                  filter: 'drop-shadow(0 0 2px rgba(0, 0, 0, 0.5))'
                }}
                className={currentPosition.isInteracting ? 'animate-pulse' : ''}
              />

              {currentPosition.isInteracting && (
                <div className="absolute -top-1 -right-1">
                  <div className="bg-yellow-400 text-xs px-1 rounded-full">
                    🔮
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default EnhancedPathVisualizer; 
