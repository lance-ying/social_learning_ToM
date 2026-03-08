import { useEffect, useState, useCallback } from 'react';
import { firebaseLogger } from '../services/FirebaseLogger';

interface BatchStatus {
  pendingEvents: number;
  activeLevels: string[];
  batchTimerActive: boolean;
  isEnabled: boolean;
}

export const useFirebaseBatching = () => {
  const [batchStatus, setBatchStatus] = useState<BatchStatus>({
    pendingEvents: 0,
    activeLevels: [],
    batchTimerActive: false,
    isEnabled: false
  });

  // Update batch status periodically
  useEffect(() => {
    const updateStatus = () => {
      const status = firebaseLogger.getBatchStatus();
      setBatchStatus({
        ...status,
        isEnabled: firebaseLogger.isLoggingEnabled()
      });
    };

    // Update immediately
    updateStatus();

    // Set up interval to update status
    const interval = setInterval(updateStatus, 2000); // Update every 2 seconds

    return () => clearInterval(interval);
  }, []);

  // Manual flush function
  const flushBatches = useCallback(async () => {
    try {
      await firebaseLogger.flushPendingBatches();
      console.log('Batches flushed manually');
    } catch (error) {
      console.error('Failed to flush batches:', error);
    }
  }, []);

  return {
    batchStatus,
    flushBatches
  };
};