"use client"
import React, { useState } from 'react';
import VisualizerModal from './VisualizerModal';

interface DebugPanelProps {
  agents: any[];
  onPathChange: (agentId: number, pathType: string) => void;
  gameState: any;
  onLevelChange?: (levelId: string) => void;
}

const DebugPanel: React.FC<DebugPanelProps> = ({ agents, onPathChange, gameState, onLevelChange }) => {
  const [showVisualizer, setShowVisualizer] = useState(false);
  
  const sortedAgents = [...agents].sort((a, b) => a.id - b.id);

  return (
    <div className="bg-gray-100 p-4 rounded-lg shadow-md">
      <h3 className="text-lg font-bold mb-4">Debug Controls</h3>
      
      <div className="mb-4">
        {sortedAgents.map(agent => (
          <div key={agent.id} className="mb-4">
            <h4 className="font-semibold mb-2">Agent {agent.id}</h4>
            <select
              className="border rounded px-2 py-1 w-full"
              value={agent.selectedPath || 'experienced1'}
              onChange={(e) => onPathChange(agent.id, e.target.value)}
            >
              <option value="experienced1">Expert Path</option>
              <option value="experienced2">Novice Path</option>
              <option value="experienced3">Expert Path_2</option>
              <option value="experienced4">Novice Path_2</option>
            </select>
          </div>
        ))}
      </div>
      
      <button
        onClick={() => setShowVisualizer(true)}
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
      >
        Open Path Visualizer
      </button>
      
      <VisualizerModal
        isOpen={showVisualizer}
        onClose={() => setShowVisualizer(false)}
        gameState={gameState}
        onPathChange={onPathChange}
        onLevelChange={onLevelChange}
      />
    </div>
  );
};

export default DebugPanel; 