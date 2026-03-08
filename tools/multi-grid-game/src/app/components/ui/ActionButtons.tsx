import React from 'react';
import { Eye } from 'lucide-react';

interface ActionButtonsProps {
  onMove: (direction: string) => void;
  onObserve: (agentId: number) => void;
  agents: Array<{ id: number; observesRemaining: number; color: string; type?: string }>;
  disabled?: boolean;
  observeCooldownRemaining?: number;
  moveCooldownRemaining?: number;
}

const ActionButtons: React.FC<ActionButtonsProps> = ({
  onMove,
  onObserve,
  agents,
  disabled = false,
  observeCooldownRemaining = 0,
  moveCooldownRemaining = 0
}) => {
  const isMoveDisabled = disabled || moveCooldownRemaining > 0;
  return (
    <div className="bg-white p-3 rounded-lg shadow-md w-full">
      <h2 className="text-2xl font-bold mb-3 text-center">Actions</h2>
      <div className="space-y-1.5">
        <button
          onClick={() => onMove('up')}
          disabled={isMoveDisabled}
          className={`w-full h-16 bg-orange-500 text-white rounded-lg hover:bg-red-600 flex flex-col items-center justify-center ${
            isMoveDisabled ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          <span className="text-4xl">↑</span>
          <span className="text-base font-semibold">Up</span>
        </button>

        <button
          onClick={() => onMove('left')}
          disabled={isMoveDisabled}
          className={`w-full h-16 bg-orange-500 text-white rounded-lg hover:bg-red-600 flex flex-col items-center justify-center ${
            isMoveDisabled ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          <span className="text-4xl">←</span>
          <span className="text-base font-semibold">Left</span>
        </button>

        <button
          onClick={() => onMove('right')}
          disabled={isMoveDisabled}
          className={`w-full h-16 bg-orange-500 text-white rounded-lg hover:bg-red-600 flex flex-col items-center justify-center ${
            isMoveDisabled ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          <span className="text-4xl">→</span>
          <span className="text-base font-semibold">Right</span>
        </button>

        <button
          onClick={() => onMove('down')}
          disabled={isMoveDisabled}
          className={`w-full h-16 bg-orange-500 text-white rounded-lg hover:bg-red-600 flex flex-col items-center justify-center ${
            isMoveDisabled ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          <span className="text-4xl">↓</span>
          <span className="text-base font-semibold">Down</span>
        </button>

        {agents
          .sort((a, b) => a.id - b.id)
          .map((agent: any) => {
            const isObserveDisabled = disabled || observeCooldownRemaining > 0;
            return (
              <button
                key={agent.id}
                onClick={() => onObserve(agent.id)}
                disabled={isObserveDisabled}
                className={`w-full h-28 ${
                  agent.color === 'blue' ? 'bg-blue-500' : 'bg-green-500'
                } text-white rounded-lg hover:opacity-90 flex items-center justify-start gap-4 px-4 ${
                  isObserveDisabled ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                <Eye size={32} />
                <div className="flex flex-col items-start justify-center flex-1">
                  <div className="flex items-center gap-1 text-sm font-semibold">
                    <img
                      src={agent.color === 'blue' ? '/icons/al.png' : '/icons/green_a.png'}
                      alt={`P${agent.id}`}
                      className="w-5 h-5"
                    />
                    <span>({agent.observesRemaining})</span>
                  </div>
                  <span className="text-lg font-bold leading-tight">
                    Observe
                  </span>
                  {agent.type && (
                    <span className="text-sm font-bold mt-0.5 text-yellow-100">
                      {agent.type}
                    </span>
                  )}
                </div>
              </button>
            );
          })}
      </div>
    </div>
  );
};

export default ActionButtons;