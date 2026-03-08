#!/usr/bin/env node

/**
 * CLI script to delete all Firebase Realtime Database data
 * Automatically backs up data before deleting
 *
 * Usage: node scripts/deleteFirebaseData.js
 */

const { initializeApp } = require('firebase/app');
const { getDatabase, ref, remove, get } = require('firebase/database');
const fs = require('fs');
const path = require('path');

// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBHKxG0TzgWhjybTei9aCPB8YcFFcRJmUI",
  authDomain: "multi-grid-game-fc79c.firebaseapp.com",
  databaseURL: "https://multi-grid-game-fc79c-default-rtdb.firebaseio.com",
  projectId: "multi-grid-game-fc79c",
  storageBucket: "multi-grid-game-fc79c.firebasestorage.app",
  messagingSenderId: "213826896928",
  appId: "1:213826896928:web:382e7e1b55b1edc1632845",
  measurementId: "G-KRHQ4Q6XPE"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getDatabase(app);

async function deleteAllData() {
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
      const filepath = path.join(process.cwd(), filename);

      // Save the data to file
      fs.writeFileSync(filepath, JSON.stringify(data, null, 2));
      console.log(`✅ Data downloaded as ${filename}`);
      console.log(`   Saved to: ${filepath}`);
    } else {
      console.log('⚠️  No data to download');
    }

    // Then delete all data
    console.log('🗑️  Deleting all data from Firebase...');
    await remove(ref(db, '/'));
    console.log('✅ All database data deleted successfully');

    process.exit(0);
  } catch (error) {
    console.error('❌ Failed to delete all data:', error);
    process.exit(1);
  }
}

// Run the deletion
deleteAllData();
