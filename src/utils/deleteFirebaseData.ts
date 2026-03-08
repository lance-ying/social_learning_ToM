import { ref, remove, get } from 'firebase/database';
import { db } from '../config/firebaseConfig';

/**
 * Utility functions to delete data from Firebase Realtime Database
 */

export interface DeleteOptions {
  // Confirm deletion to prevent accidental data loss
  confirm?: boolean;
}

/**
 * Download data as JSON file
 */
function downloadJSON(data: any, filename: string): void {
  const jsonStr = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(url);
}

/**
 * Delete all data from the database after backing it up
 * Automatically downloads all data as data_YYYY-MM-DD_HH-MM-SS.json before deleting
 */
export async function deleteAllData(options: DeleteOptions = {}): Promise<void> {
  if (!options.confirm) {
    throw new Error('Must set confirm: true to delete all data');
  }

  try {
    // First, download all data
    console.log('📥 Downloading all data...');
    const snapshot = await get(ref(db, '/'));

    if (snapshot.exists()) {
      const data = snapshot.val();

      // Create filename with current datetime
      const now = new Date();
      const datetime = now.toISOString()
        .replace(/T/, '_')
        .replace(/:/g, '-')
        .split('.')[0]; // Format: YYYY-MM-DD_HH-MM-SS

      const filename = `data_${datetime}.json`;

      // Download the data
      downloadJSON(data, filename);
      console.log(`✅ Data downloaded as ${filename}`);
    } else {
      console.log('⚠️ No data to download');
    }

    // Then delete all data
    console.log('🗑️ Deleting all data...');
    await remove(ref(db, '/'));
    console.log('✅ All database data deleted successfully');
  } catch (error) {
    console.error('❌ Failed to delete all data:', error);
    throw error;
  }
}

/**
 * Delete all users data
 */
export async function deleteAllUsers(options: DeleteOptions = {}): Promise<void> {
  if (!options.confirm) {
    throw new Error('Must set confirm: true to delete all users');
  }

  try {
    await remove(ref(db, 'users'));
    console.log('✅ All users deleted successfully');
  } catch (error) {
    console.error('❌ Failed to delete users:', error);
    throw error;
  }
}

/**
 * Delete a specific user session by ID
 */
export async function deleteUserSession(sessionId: string): Promise<void> {
  try {
    await remove(ref(db, `users/${sessionId}`));
    console.log(`✅ User session ${sessionId} deleted successfully`);
  } catch (error) {
    console.error(`❌ Failed to delete user session ${sessionId}:`, error);
    throw error;
  }
}

/**
 * Delete a specific level within a user session
 */
export async function deleteLevelData(
  sessionId: string,
  levelId: string
): Promise<void> {
  try {
    await remove(ref(db, `users/${sessionId}/levels/${levelId}`));
    console.log(`✅ Level ${levelId} deleted for session ${sessionId}`);
  } catch (error) {
    console.error(`❌ Failed to delete level ${levelId}:`, error);
    throw error;
  }
}

/**
 * Delete the session counter
 */
export async function deleteSessionCounter(): Promise<void> {
  try {
    await remove(ref(db, 'counters/sessionCounter'));
    console.log('✅ Session counter deleted successfully');
  } catch (error) {
    console.error('❌ Failed to delete session counter:', error);
    throw error;
  }
}

/**
 * Delete all counters
 */
export async function deleteAllCounters(options: DeleteOptions = {}): Promise<void> {
  if (!options.confirm) {
    throw new Error('Must set confirm: true to delete all counters');
  }

  try {
    await remove(ref(db, 'counters'));
    console.log('✅ All counters deleted successfully');
  } catch (error) {
    console.error('❌ Failed to delete counters:', error);
    throw error;
  }
}

/**
 * Get list of all user sessions
 */
export async function listAllUserSessions(): Promise<string[]> {
  try {
    const snapshot = await get(ref(db, 'users'));
    if (snapshot.exists()) {
      return Object.keys(snapshot.val());
    }
    return [];
  } catch (error) {
    console.error('❌ Failed to list user sessions:', error);
    throw error;
  }
}

/**
 * Delete old user sessions (older than specified days)
 */
export async function deleteOldSessions(daysOld: number): Promise<number> {
  try {
    const snapshot = await get(ref(db, 'users'));
    if (!snapshot.exists()) {
      console.log('No users to delete');
      return 0;
    }

    const users = snapshot.val();
    const cutoffTime = Date.now() - daysOld * 24 * 60 * 60 * 1000;
    let deletedCount = 0;

    for (const [sessionId, userData] of Object.entries(users)) {
      const startTime = (userData as any).startTime;
      if (startTime && startTime < cutoffTime) {
        await deleteUserSession(sessionId);
        deletedCount++;
      }
    }

    console.log(`✅ Deleted ${deletedCount} old sessions`);
    return deletedCount;
  } catch (error) {
    console.error('❌ Failed to delete old sessions:', error);
    throw error;
  }
}

/**
 * Get database statistics
 */
export async function getDatabaseStats(): Promise<{
  totalUsers: number;
  totalCounterValue: number | null;
  userSessions: string[];
}> {
  try {
    const [usersSnapshot, counterSnapshot] = await Promise.all([
      get(ref(db, 'users')),
      get(ref(db, 'counters/sessionCounter'))
    ]);

    const userSessions = usersSnapshot.exists()
      ? Object.keys(usersSnapshot.val())
      : [];

    return {
      totalUsers: userSessions.length,
      totalCounterValue: counterSnapshot.exists() ? counterSnapshot.val() : null,
      userSessions
    };
  } catch (error) {
    console.error('❌ Failed to get database stats:', error);
    throw error;
  }
}
