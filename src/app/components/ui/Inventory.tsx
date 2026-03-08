import React from 'react';
import { GiGemPendant } from "react-icons/gi";

interface InventoryProps {
  items: string[];
}

const Inventory: React.FC<InventoryProps> = ({ items }) => {
  const getItemStyle = (item: string) => {
    if (item?.includes('blue')) return 'bg-blue-500';
    if (item?.includes('red')) return 'bg-red-500';
    return '';
  };

  const validItems = Array.from(new Set(items.filter(item => 
    item?.includes('red') || item?.includes('blue')
  )));

  return (
    <div className="bg-white p-3 rounded-lg shadow-md w-full">
      <h2 className="text-2xl text-center font-bold mb-2 text-black">Inventory</h2>
      <div className="flex space-x-2 justify-center">
        {/* Always show exactly 2 slots */}
        <div
          className={`w-16 h-16 border-2 border-gray-300 rounded-lg flex items-center justify-center ${getItemStyle(validItems[0] || '')}`}
        >
          {validItems[0] && <span className="text-white text-2xl font-bold"><GiGemPendant /></span>}
        </div>
        <div
          className={`w-16 h-16 border-2 border-gray-300 rounded-lg flex items-center justify-center ${getItemStyle(validItems[1] || '')}`}
        >
          {validItems[1] && <span className="text-white text-2xl font-bold"><GiGemPendant /></span>}
        </div>
      </div>
    </div>
  );
};

export default Inventory;