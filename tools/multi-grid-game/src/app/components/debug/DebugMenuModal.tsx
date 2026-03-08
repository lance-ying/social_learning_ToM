"use client"
import React from 'react';

interface DebugMenuModalProps {
  onViewLevels: () => void;
  onViewUserData: () => void;
  onClose: () => void;
}

const DebugMenuModal: React.FC<DebugMenuModalProps> = ({ onViewLevels, onViewUserData, onClose }) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white p-6 rounded-lg shadow-lg max-w-md w-full">
        <h2 className="text-2xl font-bold mb-6 text-black">Debug Menu</h2>
        <div className="space-y-4">
          <button
            onClick={onViewLevels}
            className="w-full px-6 py-3 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors text-lg font-semibold"
          >
            View & Test Levels
          </button>
          <button
            onClick={onViewUserData}
            className="w-full px-6 py-3 bg-green-500 text-white rounded hover:bg-green-600 transition-colors text-lg font-semibold"
          >
            View UserData Collection
          </button>
          <button
            onClick={onClose}
            className="w-full px-6 py-3 bg-gray-300 text-black rounded hover:bg-gray-400 transition-colors text-lg font-semibold"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};

export default DebugMenuModal;
