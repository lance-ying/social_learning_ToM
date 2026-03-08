"use client"
import React from 'react';
import { Agent, Player } from '@/types/game';

interface NPCTrailProps {
  npc: Agent | Player;
  cellSize: number;
  gameSessionId?: string;
}

const NPCTrail = ({ npc, cellSize }: NPCTrailProps) => {
  const getArrowStyle = (direction: string) => {
    switch (direction) {
      case 'up': return { transform: 'rotate(0deg)' };
      case 'down': return { transform: 'rotate(180deg)' };
      case 'left': return { transform: 'rotate(270deg)' };
      case 'right': return { transform: 'rotate(90deg)' };
      default: return {};
    }
  };

  // Get movement history from agent or use fallback for player
  const getMovementHistory = () => {
    if ('movementHistory' in npc && npc.movementHistory) {
      return npc.movementHistory;
    }
    // Fallback for player or agents without movement history
    return [];
  };

  let movementHistory = getMovementHistory();

  // Handle trail reset for agents - use local state instead of mutating props
  const [shouldShowTrail, setShouldShowTrail] = React.useState(true);
  
  React.useEffect(() => {
    if ('resetTrails' in npc && npc.resetTrails) {
      // Hide trails temporarily when resetTrails is true, but don't mutate the data
      setShouldShowTrail(false);
      // Reset to show trails after a brief delay
      setTimeout(() => setShouldShowTrail(true), 150);
    }
  }, [npc.resetTrails]);

  // Determine arrow color based on entity type
  const arrowColor = 'color' in npc 
    ? (npc.color === 'blue' ? 'border-b-blue-500' : 'border-b-green-500')
    : 'border-b-red-500'; // Player gets red arrows

  // Don't render anything if trails should be hidden
  if (!shouldShowTrail) {
    return null;
  }

  return (
    <>
      {movementHistory.map((pos, index) => (
        <div
          key={`${pos.x}-${pos.y}-${pos.timestamp}-${index}`}
          className="absolute flex items-center justify-center"
          style={{
            left: `${pos.x * cellSize}px`,
            top: `${pos.y * cellSize}px`,
            width: `${cellSize}px`,
            height: `${cellSize}px`,
            opacity: 0.7 - index * 0.07,
          }}
        >
          <div
            className={`w-0 h-0 border-l-[10px] border-l-transparent border-r-[10px] border-r-transparent border-b-[20px] ${arrowColor}`}
            style={getArrowStyle(pos.direction)}
          />
        </div>
      ))}
    </>
  );
};

export default NPCTrail; 