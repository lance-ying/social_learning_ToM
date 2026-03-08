// import { collection, doc, setDoc, updateDoc, arrayUnion, serverTimestamp, addDoc } from 'firebase/firestore';
// import { db } from './firebaseConfig';

// export const createGameSession = async (userData) => {
//     const gameSessionId = `game_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
//     const gameSessionRef = doc(db, 'game_sessions', gameSessionId);
  
//     await setDoc(gameSessionRef, {
//       timestamp: serverTimestamp(),
//     });
  

//   await setDoc(gameSessionRef, {
//     timestamp: serverTimestamp(),
//   });

//   // Create the initial log entry with user data
//   const logsCollectionRef = collection(gameSessionRef, 'logs');
//   await addDoc(logsCollectionRef, {
//     type: 'USER_DATA',
//     data: {
//       prolificId: userData.prolificId,
//       age: userData.age,
//       gender: userData.gender,
//       feedback: userData.feedback
//     },
//     timestamp: serverTimestamp(),
//   });

//   return gameSessionId;
// };

// export const updateComprehensionCheck = async (gameSessionId, comprehensionData) => {
//   const gameSessionRef = doc(db, 'game_sessions', gameSessionId);
//   const logsCollectionRef = collection(gameSessionRef, 'logs');

//   await addDoc(logsCollectionRef, {
//     type: 'COMPREHENSION_CHECK',
//     data: comprehensionData,
//     timestamp: serverTimestamp(),
//   });
// };

// export const addGameLog = async (gameSessionId, logData) => {
//   const gameSessionRef = doc(db, 'game_sessions', gameSessionId);
//   const logsCollectionRef = collection(gameSessionRef, 'logs');

//   await addDoc(logsCollectionRef, {
//     ...logData,
//     timestamp: serverTimestamp(),
//   });
// };

import { doc, setDoc, updateDoc, arrayUnion, serverTimestamp, getDoc, increment, runTransaction } from 'firebase/firestore';
import { db } from './firebaseConfig';

const getAndIncrementCounter = async () => {
  const counterRef = doc(db, 'counters', 'sessionCounter');
  
  return await runTransaction(db, async (transaction) => {
    const counterDoc = await transaction.get(counterRef);
    
    if (!counterDoc.exists()) {
      transaction.set(counterRef, { count: 1 });
      return 1;
    } else {
      const newCount = counterDoc.data().count + 1;
      transaction.update(counterRef, { count: newCount });
      return newCount;
    }
  });
};

export const createGameSession = async () => {
  const sessionCounter = await getAndIncrementCounter();
  const gameSessionId = `game_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  const gameSessionRef = doc(db, 'game_sessions', gameSessionId);
  
  await setDoc(gameSessionRef, {
    createdAt: serverTimestamp(),
    logs: [],
    currentLevel: null,
    npcPathInfo: null,
    sessionNumber: sessionCounter
  });
  
  return { gameSessionId, sessionNumber: sessionCounter };
};

export const addGameLog = async (gameSessionId, logEntry) => {
  if (!gameSessionId) {
    console.warn("Cannot add game log: gameSessionId is undefined");
    return;
  }
  const gameSessionRef = doc(db, 'game_sessions', gameSessionId);
  await updateDoc(gameSessionRef, {
    logs: arrayUnion(logEntry)
  });
};

export const updateGameSession = async (gameSessionId, data) => {
  const gameSessionRef = doc(db, 'game_sessions', gameSessionId);
  await updateDoc(gameSessionRef, {
    ...data,
    updatedAt: serverTimestamp()
  });
};

export const addComprehensionCheck = async (gameSessionId, checkData) => {
  const gameSessionRef = doc(db, 'game_sessions', gameSessionId);
  const sanitizedCheckData = {
    answers: checkData.answers || {},
    passed: checkData.passed || false,
    actionLog: checkData.actionLog || [],
    failedAttempts: checkData.failedAttempts || 0,
    timestamp: serverTimestamp()
  };
  await updateDoc(gameSessionRef, {
    comprehensionCheck: sanitizedCheckData
  });
};


export const addUserInfo = async (gameSessionId, userInfo) => {
  const gameSessionRef = doc(db, 'game_sessions', gameSessionId);
  await updateDoc(gameSessionRef, {
    userInfo: {
      ...userInfo,
      timestamp: serverTimestamp()
    }
  });
};

export const updateSessionWithLevelInfo = async (gameSessionId, levelName, npcPathInfo) => {
  const gameSessionRef = doc(db, 'game_sessions', gameSessionId);
  await updateDoc(gameSessionRef, {
    currentLevel: levelName,
    npcPathInfo: npcPathInfo
  });
};

export const getComprehensionCheck = async (gameSessionId) => {
  const gameSessionRef = doc(db, 'game_sessions', gameSessionId);
  const gameSessionDoc = await getDoc(gameSessionRef);
  if (gameSessionDoc.exists()) {
    return gameSessionDoc.data().comprehensionCheck || { failedAttempts: 0 };
  }
  return { failedAttempts: 0 };
};

export const incrementFailedAttempts = async (gameSessionId) => {
  const gameSessionRef = doc(db, 'game_sessions', gameSessionId);
  
  return runTransaction(db, async (transaction) => {
    const gameSessionDoc = await transaction.get(gameSessionRef);
    
    if (!gameSessionDoc.exists()) {
      throw new Error("Document does not exist!");
    }

    const currentFailedAttempts = (gameSessionDoc.data().comprehensionCheck?.failedAttempts || 0) + 1;
    
    transaction.update(gameSessionRef, {
      'comprehensionCheck.failedAttempts': currentFailedAttempts
    });

    return currentFailedAttempts;
  });
};