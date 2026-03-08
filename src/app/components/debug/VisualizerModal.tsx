"use client"
import React, { useEffect, useRef } from 'react';
import { X } from 'lucide-react';
import EnhancedPathVisualizer from './EnhancedPathVisualizer';

interface VisualizerModalProps {
  isOpen: boolean;
  onClose: () => void;
  gameState: any;
  onPathChange: (agentId: number, pathType: string) => void;
  onLevelChange?: (levelId: string) => void;
}

const VisualizerModal: React.FC<VisualizerModalProps> = ({ 
  isOpen, 
  onClose, 
  gameState, 
  onPathChange,
  onLevelChange
}) => {
  const modalRef = useRef<HTMLDivElement>(null);

  // Close modal when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(event.target as Node)) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen, onClose]);

  // Prevent scrolling when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'auto';
    }
    
    return () => {
      document.body.style.overflow = 'auto';
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <div 
        ref={modalRef}
        className="bg-white rounded-lg shadow-xl p-6 max-w-5xl max-h-[90vh] overflow-auto"
      >
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold">Agent Path Visualizer</h2>
          <button 
            onClick={onClose}
            className="p-1 rounded-full hover:bg-gray-200 transition-colors"
          >
            <X size={24} />
          </button>
        </div>
        
        <EnhancedPathVisualizer 
          gameState={gameState} 
          onPathChange={onPathChange}
          onLevelChange={onLevelChange}
        />
      </div>
    </div>
  );
};

export default VisualizerModal; 