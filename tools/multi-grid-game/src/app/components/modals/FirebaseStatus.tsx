"use client"
import React, { useState, useEffect } from 'react';
import { firebaseLogger } from '@/services/FirebaseLogger';

interface FirebaseStatusProps {
  className?: string;
}

export const FirebaseStatus: React.FC<FirebaseStatusProps> = ({ className = '' }) => {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLogging, setIsLogging] = useState(false);
  const [eventCount, setEventCount] = useState(0);

  useEffect(() => {
    setSessionId(firebaseLogger.getSessionId());
    setIsLogging(firebaseLogger.isLoggingEnabled());
    
    // Update status every few seconds
    const interval = setInterval(() => {
      setSessionId(firebaseLogger.getSessionId());
      setIsLogging(firebaseLogger.isLoggingEnabled());
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  if (!isLogging) {
    return (
      <div className={`text-sm text-gray-500 ${className}`}>
        📊 Firebase: Disabled
      </div>
    );
  }

  return (
    <div className={`text-sm text-green-600 ${className}`}>
      <div className="flex items-center gap-2">
        <span>🔥 Firebase: Active</span>
        {sessionId && (
          <span className="text-xs bg-green-100 px-2 py-1 rounded">
            {sessionId.split('_')[1]?.substring(0, 6)}...
          </span>
        )}
      </div>
    </div>
  );
};

export default FirebaseStatus;