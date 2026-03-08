"use client"
import React, { useMemo, useEffect } from 'react';
import { GameState } from '@/types/game';
import { PiTreasureChestBold } from "react-icons/pi";
import { GiNestedHexagons } from "react-icons/gi";
import NPCTrail from '../ui/NPCTrail';
import { getMapDimensions } from '@/data/mapData';

interface GameGridProps {
  state: GameState;
  asciiMap?: string;
  activeInteraction?: {
    wizardX: number;
    wizardY: number;
    agentId: number | null;
    wizardColor: string;
    timestamp: number;
  } | null;
}

const GameGrid = React.memo(({ state, asciiMap, activeInteraction }: GameGridProps) => {
  const CELL_SIZE = 42;

  useEffect(() => {
    if (activeInteraction) {
      console.log("GameGrid received activeInteraction:", activeInteraction);
    }
  }, [activeInteraction]);

  // Calculate grid dimensions from map data or use defaults
  const { gridWidth, gridHeight } = useMemo(() => {
    if (asciiMap) {
      const { width, height } = getMapDimensions(asciiMap);
      return { gridWidth: width, gridHeight: height };
    }
    
    // Fallback: calculate from existing blocks/entities
    const allX = [...state.blocks.map(b => b.x), ...state.agents.map(a => a.x), state.player.x];
    const allY = [...state.blocks.map(b => b.y), ...state.agents.map(a => a.y), state.player.y];
    const maxX = Math.max(...allX, 10);
    const maxY = Math.max(...allY, 11);
    
    return { gridWidth: maxX + 1, gridHeight: maxY + 1 };
  }, [asciiMap, state.blocks, state.agents, state.player]);
  
  const containerWidth = gridWidth * CELL_SIZE;
  const containerHeight = gridHeight * CELL_SIZE;

  return (
    <>
      <style jsx>{`
        @keyframes bounce {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.2); }
        }
        @keyframes agent-lean {
          0%, 100% { transform: translate(0, 0) scale(1); }
          50% { transform: translate(var(--lean-x), var(--lean-y)) scale(1.1); }
        }
        @keyframes sparkle-float {
          0% {
            opacity: 1;
            transform: translate(-5px, -5px) scale(1);
          }
          100% {
            opacity: 0;
            transform: translate(var(--tx), var(--ty)) scale(0.3);
          }
        }
        .bounce-animation {
          animation: bounce 0.5s ease-in-out;
        }
        .agent-lean-animation {
          animation: agent-lean 0.5s ease-in-out;
        }
        .sparkle {
          animation: sparkle-float 0.5s ease-out forwards;
          z-index: 1000;
        }
      `}</style>
      <div
        className="relative bg-white rounded-lg border border-gray-200"
        style={{ width: `${containerWidth}px`, height: `${containerHeight}px` }}
      >
      {/* Base grid */}
      {[...Array(gridHeight)].map((_, y) =>
        [...Array(gridWidth)].map((_, x) => {
          const isBlock = state.blocks.some(block => block.x === x && block.y === y);
          return (
            <div
              key={`${x}-${y}`}
              className={`absolute ${
                isBlock ? 'bg-gray-800' : 'bg-white'
              }`}
              style={{
                left: `${x * CELL_SIZE}px`,
                top: `${y * CELL_SIZE}px`,
                width: `${CELL_SIZE}px`,
                height: `${CELL_SIZE}px`,
                // border: '0.1px solid rgba(229, 231, 235, 0.5)' // Thinner white line
              }}
            />
          );
        })
      )}

      {/* Player Trail */}
      <NPCTrail 
        key={`player-trail-${state.gameSessionId}`} 
        npc={state.player} 
        cellSize={CELL_SIZE} 
        gameSessionId={state.gameSessionId}
      />

      {/* NPC Trails */}
      {state.agents.map((agent) => (
        <NPCTrail 
          key={`npc-trail-${agent.id}-${state.gameSessionId}`} 
          npc={agent} 
          cellSize={CELL_SIZE} 
          gameSessionId={state.gameSessionId}
        />
      ))}

      {/* Barriers */}
      {state.barriers.map((barrier, i) => (
        <div
          key={`barrier-${i}`}
          className="absolute flex items-center justify-center"
          style={{
            left: `${barrier.x * CELL_SIZE}px`,
            top: `${barrier.y * CELL_SIZE}px`,
            width: `${CELL_SIZE}px`,
            height: `${CELL_SIZE}px`,
          }}
        >
          <GiNestedHexagons 
            className={`${
              barrier.requiredItems.includes('blueAmulet') ? 'text-blue-500' : 'text-red-500'
            }`} 
            style={{ fontSize: `${CELL_SIZE * 1}px` }} // Scale icon to 60% of cell size
          />
        </div>
      ))}

      {/* Treasure Pots */}
      {state.treasurePots.map((pot, i) => (
        <div
          key={`pot-${i}`}
          className="absolute flex items-center justify-center"
          style={{
            left: `${pot.x * CELL_SIZE}px`,
            top: `${pot.y * CELL_SIZE}px`,
            width: `${CELL_SIZE}px`,
            height: `${CELL_SIZE}px`,
          }}
        >
          <div className="relative">
            <PiTreasureChestBold 
              className="text-yellow-500" 
              style={{ fontSize: `${CELL_SIZE * 1}px` }} // Scale icon to 60% of cell size
            />
            <div className="absolute inset-0 flex items-center justify-center text-lg font-bold text-black">
              {pot.label}
            </div>
          </div>
        </div>
      ))}

      {/* Wizards */}
      {state.wizards.map((wizard, i) => {
        const isInteracting = activeInteraction &&
          activeInteraction.wizardX === wizard.x &&
          activeInteraction.wizardY === wizard.y;
        return (
          <div
            key={`wizard-${i}`}
            className={`absolute flex items-center justify-center ${isInteracting ? 'bounce-animation' : ''}`}
            style={{
              left: `${wizard.x * CELL_SIZE}px`,
              top: `${wizard.y * CELL_SIZE}px`,
              width: `${CELL_SIZE}px`,
              height: `${CELL_SIZE}px`,
            }}
          >
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
      })}

      {/* Player */}
      {(() => {
        const isPlayerInteracting = activeInteraction && activeInteraction.agentId === null;
        let leanX = 0;
        let leanY = 0;
        if (isPlayerInteracting && activeInteraction) {
          const dx = activeInteraction.wizardX - state.player.x;
          const dy = activeInteraction.wizardY - state.player.y;
          leanX = dx * CELL_SIZE * 0.2;
          leanY = dy * CELL_SIZE * 0.2;
        }

        return (
          <div
            className={`absolute flex items-center justify-center transition-all duration-200 ${isPlayerInteracting ? 'agent-lean-animation' : ''}`}
            style={{
              left: `${state.player.x * CELL_SIZE}px`,
              top: `${state.player.y * CELL_SIZE}px`,
              width: `${CELL_SIZE}px`,
              height: `${CELL_SIZE}px`,
              // @ts-ignore
              '--lean-x': `${leanX}px`,
              '--lean-y': `${leanY}px`,
            }}
          >
            <img
              src="/icons/bl.png"
              alt="Player"
              style={{
                width: `${CELL_SIZE * .8}px`,
                height: `${CELL_SIZE * .9}px`
              }}
            />
            {/* Main Player Label */}
            <div
              style={{
                position: 'absolute',
                bottom: '-6px',
                left: '50%',
                transform: 'translateX(-50%)',
                backgroundColor: 'rgba(0, 0, 0, 0.75)',
                color: 'white',
                fontSize: '10px',
                fontWeight: 'bold',
                padding: '1px 5px',
                borderRadius: '3px',
                lineHeight: '1',
                border: '1px solid rgba(255, 255, 255, 0.3)',
                pointerEvents: 'none',
              }}
            >
              M
            </div>
          </div>
        );
      })()}

      {/* Agents */}
      {state.agents.map((agent) => {
        const isInteracting = activeInteraction && activeInteraction.agentId === agent.id;

        // Calculate lean direction toward wizard
        let leanX = 0;
        let leanY = 0;
        if (isInteracting && activeInteraction) {
          const dx = activeInteraction.wizardX - agent.x;
          const dy = activeInteraction.wizardY - agent.y;
          leanX = dx * CELL_SIZE * 0.2; // 20% of cell toward wizard
          leanY = dy * CELL_SIZE * 0.2;
        }

        // Determine agent label (N for Novice, E for Expert)
        const agentLabel = agent.type === 'Expert' ? 'E' : agent.type === 'Novice' ? 'N' : '';

        return (
          <div
            key={agent.id}
            className={`absolute flex items-center justify-center transition-all duration-200 ${isInteracting ? 'agent-lean-animation' : ''}`}
            style={{
              left: `${agent.x * CELL_SIZE}px`,
              top: `${agent.y * CELL_SIZE}px`,
              width: `${CELL_SIZE}px`,
              height: `${CELL_SIZE}px`,
              // @ts-ignore
              '--lean-x': `${leanX}px`,
              '--lean-y': `${leanY}px`,
            }}
          >
            <img
              src={agent.color === 'blue' ? '/icons/al.png' : '/icons/green_a.png'}
              alt={agent.color === 'blue' ? `Player ${agent.id}` : `Player ${agent.id}`}
              style={{
                width: `${CELL_SIZE * 0.6}px`,
                height: `${CELL_SIZE * 0.7}px`
              }}
            />
            {/* Agent Type Label */}
            {agentLabel && (
              <div
                style={{
                  position: 'absolute',
                  bottom: '-6px',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  backgroundColor: 'rgba(0, 0, 0, 0.75)',
                  color: 'white',
                  fontSize: '10px',
                  fontWeight: 'bold',
                  padding: '1px 5px',
                  borderRadius: '3px',
                  lineHeight: '1',
                  border: '1px solid rgba(255, 255, 255, 0.3)',
                  pointerEvents: 'none',
                }}
              >
                {agentLabel}
              </div>
            )}
          </div>
        );
      })}

      {/* Sparkle Particles for Wizard Interactions */}
      {activeInteraction && (() => {
        console.log("Rendering sparkles at wizard position:", activeInteraction.wizardX, activeInteraction.wizardY, "color:", activeInteraction.wizardColor);

        // Determine sparkle colors based on wizard color
        const isRed = activeInteraction.wizardColor.includes('red');
        const sparkleGradient = isRed
          ? 'radial-gradient(circle, #ff4444, #ff6b6b)'
          : 'radial-gradient(circle, #4444ff, #6b6bff)';
        const sparkleShadow = isRed
          ? '0 0 12px #ff4444, 0 0 20px rgba(255, 68, 68, 0.6)'
          : '0 0 12px #4444ff, 0 0 20px rgba(68, 68, 255, 0.6)';

        return (
          <>
            {[...Array(6)].map((_, i) => {
              const angle = (i * Math.PI * 2) / 6;
              const distance = 35;
              const tx = Math.cos(angle) * distance;
              const ty = Math.sin(angle) * distance;
              return (
                <div
                  key={`sparkle-${i}-${activeInteraction.timestamp}`}
                  className="sparkle absolute pointer-events-none"
                  style={{
                    left: `${activeInteraction.wizardX * CELL_SIZE + CELL_SIZE / 2 - 5}px`,
                    top: `${activeInteraction.wizardY * CELL_SIZE + CELL_SIZE / 2 - 5}px`,
                    width: '10px',
                    height: '10px',
                    background: sparkleGradient,
                    borderRadius: '50%',
                    boxShadow: sparkleShadow,
                    border: '1.5px solid rgba(255, 255, 255, 0.8)',
                    zIndex: 1000,
                    // @ts-ignore
                    '--tx': `${tx}px`,
                    '--ty': `${ty}px`,
                  }}
                />
              );
            })}
          </>
        );
      })()}
      </div>
    </>
  );
});

export default GameGrid;