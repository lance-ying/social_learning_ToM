"use client"
import React, { useState } from 'react';
import { ChevronUp, ChevronDown } from 'lucide-react';
import { GiNestedHexagons } from "react-icons/gi";
import { PiTreasureChestBold } from "react-icons/pi";
import { getDisplayScore } from '@/utils/scoreDisplay';
import { EXPERIMENT_TYPE } from '@/app/components/game-flow/introductions/config';

interface GameInfoProps {
    goalType: string;
    points: number;
    otherPlayers: Array<{
      id: number;
      type: string;
      color: string;
    }>;
    isDevMode?: boolean;
  }

  const GameInfo: React.FC<GameInfoProps> = ({ goalType, points, otherPlayers, isDevMode = false }) => {
    const [isExpanded, setIsExpanded] = useState(false);

    const legendItems = [
      { 
        icon: <div className="w-5 h-5 bg-gray-800"></div>,
        label: 'Wall' 
      },
      { 
        icon: <GiNestedHexagons className="w-5 h-5 text-blue-500" />,
        label: 'Blue Barrier' 
      },
      { 
        icon: <GiNestedHexagons className="w-5 h-5 text-red-500" />,
        label: 'Red Barrier' 
      },
      { 
        icon: <PiTreasureChestBold className="w-5 h-5 text-yellow-500" />,
        label: 'Treasure' 
      },
      { 
        icon: <img src="/icons/d.png" alt="Blue Wizard" className="w-5 h-5" />,
        label: 'Blue Wizard' 
      },
      { 
        icon: <img src="/icons/f.png" alt="Red Wizard" className="w-5 h-5" />,
        label: 'Red Wizard' 
      },
      { 
        icon: <img src="/icons/bl.png" alt="You" className="w-4 h-5" />,
        label: 'You' 
      },
      { 
        icon: <img src="/icons/al.png" alt="Player 1" className="w-3 h-4" />,
        label: 'Player 1' 
      },
    ];

    return (
      <div className="bg-white p-4 rounded-lg shadow-md w-full max-w-xs">
        <h2 className="text-3xl font-bold mb-4 text-center">Game Info</h2>
        <div className="space-y-4">
          <div className="bg-gray-100 p-3 rounded-lg">
            <span className="text-xl font-semibold block mb-1 text-center">Your Goal:</span>
            <div className="flex flex-col items-center">
              <span className="text-3xl font-bold text-yellow-600">Treasure</span>
              <span className="text-3xl font-bold text-yellow-600">{goalType}</span>
            </div>
          </div>
          <div className="bg-gray-100 p-3 rounded-lg">
            <span className="text-xl font-semibold block mb-1">
              {isDevMode ? 'Points Used:' : 'Your Points:'}
            </span>
            <span className={`text-6xl font-bold ${isDevMode ? 'text-blue-600' : 'text-green-600'}`}>
              {isDevMode ? points : getDisplayScore(points)}
            </span>
          </div>

          {otherPlayers.map((player, index) => (
            <div key={index} className="bg-gray-100 p-3 rounded-lg">
              <span className="text-xl font-semibold block mb-1">Player {player.id}:</span>
              <span className={`text-3xl font-bold ${player.color === 'blue' ? 'text-blue-600' : 'text-green-600'}`}>
                {player.type}
              </span>
            </div>
          ))}

          <div className="bg-gray-100 p-3 rounded-lg">
            <button 
              className="w-full flex justify-between items-center font-bold text-xl"
              onClick={() => setIsExpanded(!isExpanded)}
            >
              Legend
              {isExpanded ? <ChevronUp size={24} /> : <ChevronDown size={24} />}
            </button>
            
            {isExpanded && (
              <div className="mt-3 space-y-2">
                {legendItems.map((item, index) => (
                  <div key={index} className="flex items-center text-sm">
                    <div className="w-8 h-8 flex items-center justify-center mr-2">
                      {item.icon}
                    </div>
                    <span>{item.label}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

export default GameInfo