import { initializeApp } from 'firebase/app';
import { getDatabase } from 'firebase/database';

// Your web app's Firebase configuration
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
export const db = getDatabase(app);