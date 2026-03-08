import React, { useState, useEffect, useCallback, useRef } from 'react';
import GridGame from './components/GridGame/GridGame';
import GameIntroduction from './components/GridGame/GameIntroduction';
import ComprehensionCheck from './components/GridGame/ComprehensionCheck';
import UserInfoForm from './components/GridGame/UserInfoForm';
import TestModeGame from './components/GridGame/TestModeGame';
import DebugPasswordModal from './components/GridGame/DebugPasswordModal';
import GamePathVisualization from './components/GridGame/GamePathVisualization';
import { 
  createGameSession, 
  updateGameSession,
  addGameLog
} from './components/GridGame/firestoreHelpers'
 
const App = () => {
  const [gameState, setGameState] = useState('introduction');
  const [userInfo, setUserInfo] = useState(null);
  const [debugMode, setDebugMode] = useState(false);
  const [showPasswordModal, setShowPasswordModal] = useState(false);
  const [gameSessionId, setGameSessionId] = useState(null);
  const [currentLevelIndex, setCurrentLevelIndex] = useState(0);
  const [sessionNumber, setSessionNumber] = useState(null);
  const [currentSequence, setCurrentSequence] = useState([]);
  const [currentLevelInfo, setCurrentLevelInfo] = useState(null);
  const [tutorialCompleted, setTutorialCompleted] = useState(false);
  const [totalPoints, setTotalPoints] = useState(0);
  const [lastTrialPoints, setLastTrialPoints] = useState(0);

//   const sequences = [
//     ['s111_1_1', 's112_2_1', 's441_1_1', 's531_1_2', 's351_1_1', 's331_2_1', 's362_1_1', 's411_1_1', 's341_1_2', 's541_1_2', 's521_2_1', 's431_2_1'],
//     ['s111_2_2', 's112_1_2', 's342_1_2', 's431_1_1', 's351_2_1', 's211_2_1', 's221_1_2', 's331_1_2', 's532_1_2', 's442_1_1', 's541_1_1', 's361_1_2'],
//     ['s111_1_2', 's112_2_2', 's511_2_1', 's321_1_1', 's531_1_1', 's352_1_1', 's431_1_2', 's332_2_1', 's541_2_1', 's342_2_1', 's441_1_2', 's362_1_1'],
//     ['s111_2_1', 's112_1_1', 's531_1_1', 's542_1_2', 's352_1_2', 's432_2_1', 's441_2_1', 's332_1_2', 's311_2_1', 's221_2_1', 's342_1_1', 's361_2_1'],
//     ['s111_1_1', 's112_2_1', 's511_1_1', 's361_1_1', 's352_1_2', 's341_2_1', 's542_1_1', 's332_2_1', 's532_1_1', 's441_1_1', 's421_1_2', 's432_1_1'],
//     ['s111_2_2', 's112_1_2', 's442_2_1', 's331_1_1', 's321_1_2', 's542_2_1', 's432_1_2', 's531_2_1', 's362_2_1', 's342_1_2', 's311_1_1', 's351_1_2'],
//     ['s111_1_2', 's112_2_2', 's351_2_1', 's211_1_1', 's541_1_2', 's341_2_1', 's532_1_2', 's442_1_1', 's521_1_1', 's431_1_1', 's332_1_2', 's362_1_2'],
//     ['s111_2_1', 's112_1_1', 's531_2_1', 's341_1_2', 's361_2_1', 's432_1_1', 's521_1_2', 's311_1_2', 's352_2_1', 's332_1_1', 's442_2_1', 's541_1_1'],
//     ['s111_1_1', 's112_2_1', 's541_2_1', 's341_1_1', 's221_1_1', 's211_1_2', 's361_1_2', 's432_2_1', 's351_1_2', 's442_1_2', 's331_1_1', 's532_1_1'],
//     ['s111_2_2', 's112_1_2', 's352_1_1', 's362_1_2', 's441_1_2', 's532_2_1', 's431_1_2', 's341_1_1', 's331_1_2', 's542_1_2', 's321_2_1', 's411_2_1'],
//     ['s111_1_2', 's112_2_2', 's442_1_2', 's542_1_1', 's342_1_1', 's352_2_1', 's421_1_1', 's331_2_1', 's431_2_1', 's531_1_2', 's411_1_2', 's362_2_1'],
//     ['s111_2_1', 's112_1_1', 's332_1_1', 's511_1_2', 's542_2_1', 's361_1_1', 's532_2_1', 's421_2_1', 's432_1_2', 's351_1_1', 's342_2_1', 's441_2_1']
// ]

// const sequences = [
//   ['s111_1_1', 's112_2_1', 's441_1_1', 's531_2_1', 's351_1_1', 's331_1_2', 's362_1_1', 's411_1_1', 's341_2_1', 's541_2_1', 's521_1_2', 's431_1_2'],
//   ['s111_2_2', 's112_1_2', 's342_2_1', 's431_1_1', 's351_1_2', 's211_1_2', 's221_2_1', 's331_2_1', 's532_2_1', 's442_1_1', 's541_1_1', 's361_2_1'],
//   ['s111_1_2', 's112_2_2', 's511_1_2', 's321_1_1', 's531_1_1', 's352_1_1', 's431_2_1', 's332_1_2', 's541_1_2', 's342_1_2', 's441_2_1', 's362_1_1'],
//   ['s111_2_1', 's112_1_1', 's531_1_1', 's542_2_1', 's352_2_1', 's432_1_2', 's441_1_2', 's332_2_1', 's311_1_1', 's221_1_2', 's342_1_1', 's361_1_2'],
//   ['s111_1_1', 's112_2_1', 's511_1_1', 's361_1_1', 's352_2_1', 's341_1_2', 's542_1_1', 's332_1_2', 's532_1_1', 's441_1_1', 's421_2_1', 's432_1_1'],
//   ['s111_2_2', 's112_1_2', 's442_1_2', 's331_1_1', 's321_2_1', 's542_1_2', 's432_2_1', 's531_1_2', 's362_1_2', 's342_2_1', 's311_1_1', 's351_2_1'],
//   ['s111_1_2', 's112_2_2', 's351_1_2', 's211_1_1', 's541_2_1', 's341_1_2', 's532_2_1', 's442_1_1', 's521_1_1', 's431_1_1', 's332_2_1', 's362_2_1'],
//   ['s111_2_1', 's112_1_1', 's531_1_2', 's341_2_1', 's361_1_2', 's432_1_1', 's521_2_1', 's311_1_1', 's352_1_2', 's332_1_1', 's442_1_2', 's541_1_1'],
//   ['s111_1_1', 's112_2_1', 's541_1_2', 's341_1_1', 's221_1_1', 's211_2_1', 's361_2_1', 's432_1_2', 's351_2_1', 's442_2_1', 's331_1_1', 's532_1_1'],
//   ['s111_2_2', 's112_1_2', 's352_1_1', 's362_2_1', 's441_2_1', 's532_1_2', 's431_2_1', 's341_1_1', 's331_2_1', 's542_2_1', 's321_1_2', 's411_1_1'],
//   ['s111_1_2', 's112_2_2', 's442_2_1', 's542_1_1', 's342_1_1', 's352_1_2', 's421_1_1', 's331_1_2', 's431_1_2', 's531_2_1', 's411_2_1', 's362_1_2'],
//   ['s111_2_1', 's112_1_1', 's332_1_1', 's511_2_1', 's542_1_2', 's361_1_1', 's532_1_2', 's421_1_2', 's432_2_1', 's351_1_1', 's342_1_2', 's441_1_2'],
//   ['s111_1_1', 's112_2_1', 's441_1_1', 's531_2_1', 's351_1_1', 's331_1_2', 's362_1_1', 's411_1_1', 's341_2_1', 's541_2_1', 's521_1_2', 's431_1_2'],
//   ['s111_2_2', 's112_1_2', 's342_2_1', 's431_1_1', 's351_1_2', 's211_1_2', 's221_2_1', 's331_2_1', 's532_2_1', 's442_1_1', 's541_1_1', 's361_2_1'],
//   ['s111_1_2', 's112_2_2', 's511_1_1', 's321_1_1', 's531_1_1', 's352_1_1', 's431_2_1', 's332_1_2', 's541_1_2', 's342_1_2', 's441_2_1', 's362_1_1'],
//   ['s111_2_1', 's112_1_1', 's531_1_1', 's542_2_1', 's352_2_1', 's432_1_2', 's441_1_2', 's332_2_1', 's311_1_2', 's221_1_2', 's342_1_1', 's361_1_2'],
//   ['s111_1_1', 's112_2_1', 's511_1_1', 's361_1_1', 's352_2_1', 's341_1_2', 's542_1_1', 's332_1_2', 's532_1_1', 's441_1_1', 's421_2_1', 's432_1_1'],
//   ['s111_2_2', 's112_1_2', 's442_1_2', 's331_1_1', 's321_2_1', 's542_1_2', 's432_2_1', 's531_1_2', 's362_1_2', 's342_2_1', 's311_1_1', 's351_2_1'],
//   ['s111_1_2', 's112_2_2', 's351_1_2', 's211_1_1', 's541_2_1', 's341_1_2', 's532_2_1', 's442_1_1', 's521_1_1', 's431_1_1', 's332_2_1', 's362_2_1'],
//   ['s111_2_1', 's112_1_1', 's531_1_2', 's341_2_1', 's361_1_2', 's432_1_1', 's521_2_1', 's311_2_1', 's352_1_2', 's332_1_1', 's442_1_2', 's541_1_1'],
//   ['s111_1_1', 's112_2_1', 's541_1_2', 's341_1_1', 's221_1_1', 's211_2_1', 's361_2_1', 's432_1_2', 's351_2_1', 's442_2_1', 's331_1_1', 's532_1_1'],
//   ['s111_2_2', 's112_1_2', 's352_1_1', 's362_2_1', 's441_2_1', 's532_1_2', 's431_2_1', 's341_1_1', 's331_2_1', 's542_2_1', 's321_1_2', 's411_1_2'],
//   ['s111_1_2', 's112_2_2', 's442_2_1', 's542_1_1', 's342_1_1', 's352_1_2', 's421_1_1', 's331_1_2', 's431_1_2', 's531_2_1', 's411_1_1', 's362_1_2'],
//   ['s111_2_1', 's112_1_1', 's332_1_1', 's511_1_1', 's542_1_2', 's361_1_1', 's532_1_2', 's421_1_2', 's432_2_1', 's351_1_1', 's342_1_2', 's441_1_2'],
//   ['s111_1_1', 's112_2_1', 's441_1_1', 's531_2_1', 's351_1_1', 's331_1_2', 's362_1_1', 's411_1_1', 's341_2_1', 's541_2_1', 's521_1_2', 's431_1_2'],
//   ['s111_2_2', 's112_1_2', 's342_2_1', 's431_1_1', 's351_1_2', 's211_1_1', 's221_2_1', 's331_2_1', 's532_2_1', 's442_1_1', 's541_1_1', 's361_2_1'],
//   ['s111_1_2', 's112_2_2', 's511_1_2', 's321_1_1', 's531_1_1', 's352_1_1', 's431_2_1', 's332_1_2', 's541_1_2', 's342_1_2', 's441_2_1', 's362_1_1'],
//   ['s111_2_1', 's112_1_1', 's531_1_1', 's542_2_1', 's352_2_1', 's432_1_2', 's441_1_2', 's332_2_1', 's311_1_2', 's221_1_2', 's342_1_1', 's361_1_2'],
//   ['s111_1_1', 's112_2_1', 's511_1_1', 's361_1_1', 's352_2_1', 's341_1_2', 's542_1_1', 's332_1_2', 's532_1_1', 's441_1_1', 's421_2_1', 's432_1_1'],
//   ['s111_2_2', 's112_1_2', 's442_1_2', 's331_1_1', 's321_2_1', 's542_1_2', 's432_2_1', 's531_1_2', 's362_1_2', 's342_2_1', 's311_1_1', 's351_2_1'],
//   ['s111_1_2', 's112_2_2', 's351_1_2', 's211_1_1', 's541_2_1', 's341_1_2', 's532_2_1', 's442_1_1', 's521_1_1', 's431_1_1', 's332_2_1', 's362_2_1'],
//   ['s111_2_1', 's112_1_1', 's531_1_2', 's341_2_1', 's361_1_2', 's432_1_1', 's521_2_1', 's311_2_1', 's352_1_2', 's332_1_1', 's442_1_2', 's541_1_1'],
//   ['s111_1_1', 's112_2_1', 's541_1_2', 's341_1_1', 's221_1_1', 's211_1_1', 's361_2_1', 's432_1_2', 's351_2_1', 's442_2_1', 's331_1_1', 's532_1_1'],
//   ['s111_2_2', 's112_1_2', 's352_1_1', 's362_2_1', 's441_2_1', 's532_1_2', 's431_2_1', 's341_1_1', 's331_2_1', 's542_2_1', 's321_1_2', 's411_1_2'],
//   ['s111_1_2', 's112_2_2', 's442_2_1', 's542_1_1', 's342_1_1', 's352_1_2', 's421_1_1', 's331_1_2', 's431_1_2', 's531_2_1', 's411_2_1', 's362_1_2'],
//   ['s111_2_1', 's112_1_1', 's332_1_1', 's511_2_1', 's542_1_2', 's361_1_1', 's532_1_2', 's421_1_2', 's432_2_1', 's351_1_1', 's342_1_2', 's441_1_2']
// ]

// const sequences = [
//   ['s441_1_1', 's531_1_1', 's351_1_1', 's331_1_2', 's362_1_1', 's411_1_1', 's341_2_1', 's541_2_1', 's521_1_2', 's431_1_2'],
//   ['s342_2_1', 's431_1_1', 's351_1_2', 's211_1_2', 's221_1_1', 's331_2_1', 's532_1_1', 's442_1_1', 's541_1_1', 's361_1_1'],
//   ['s511_1_2', 's321_1_1', 's531_1_1', 's352_1_1', 's431_1_1', 's332_1_2', 's541_1_2', 's342_1_2', 's441_1_1', 's362_1_1'],
//   ['s531_1_1', 's542_2_1', 's352_1_1', 's432_1_2', 's441_1_2', 's332_2_1', 's311_1_2', 's221_1_2', 's342_1_1', 's361_1_2'],
//   ['s511_1_1', 's361_1_1', 's352_1_1', 's341_1_2', 's542_1_1', 's332_1_2', 's532_1_1', 's441_1_1', 's421_1_1', 's432_1_1'],
//   ['s442_1_2', 's331_1_1', 's321_1_1', 's542_1_2', 's432_1_1', 's531_1_2', 's362_1_2', 's342_2_1', 's311_1_1', 's351_1_1'],
//   ['s351_1_2', 's211_1_1', 's541_2_1', 's341_1_2', 's532_1_1', 's442_1_1', 's521_1_1', 's431_1_1', 's332_2_1', 's362_1_1'],
//   ['s531_1_2', 's341_2_1', 's361_1_2', 's432_1_1', 's521_1_1', 's311_1_1', 's352_1_2', 's332_1_1', 's442_1_2', 's541_1_1'],
//   ['s541_1_2', 's341_1_1', 's221_1_1', 's211_1_1', 's361_1_1', 's432_1_2', 's351_1_1', 's442_1_1', 's331_1_1', 's532_1_1'],
//   ['s352_1_1', 's362_1_1', 's441_1_1', 's532_1_2', 's431_1_1', 's341_1_1', 's331_2_1', 's542_2_1', 's321_1_2', 's411_1_2'],
//   ['s442_1_1', 's542_1_1', 's342_1_1', 's352_1_2', 's421_1_1', 's331_1_2', 's431_1_2', 's531_1_1', 's411_1_1', 's362_1_2'],
//   ['s332_1_1', 's511_1_1', 's542_1_2', 's361_1_1', 's532_1_2', 's421_1_2', 's432_1_1', 's351_1_1', 's342_1_2', 's441_1_2']
// ]

// const sequences = [
//   ['s111_1_1', 's112_2_1', 's441_1_1', 's531_1_1', 's351_1_1', 's331_1_2', 's362_1_1', 's411_1_1', 's341_2_1', 's541_2_1', 's521_1_2', 's431_1_2'],
//   ['s111_1_1', 's112_2_1', 's342_2_1', 's431_1_1', 's351_1_2', 's211_1_2', 's221_1_1', 's331_2_1', 's532_1_1', 's442_1_1', 's541_1_1', 's361_1_1'],
//   ['s111_1_1', 's112_2_1', 's511_1_2', 's321_1_1', 's531_1_1', 's352_1_1', 's431_1_1', 's332_1_2', 's541_1_2', 's342_1_2', 's441_1_1', 's362_1_1'],
//   ['s111_1_1', 's112_2_1', 's531_1_1', 's542_2_1', 's352_1_1', 's432_1_2', 's441_1_2', 's332_2_1', 's311_1_2', 's221_1_2', 's342_1_1', 's361_1_2'],
//   ['s111_1_1', 's112_2_1', 's511_1_1', 's361_1_1', 's352_1_1', 's341_1_2', 's542_1_1', 's332_1_2', 's532_1_1', 's441_1_1', 's421_1_1', 's432_1_1'],
//   ['s111_1_1', 's112_2_1', 's442_1_2', 's331_1_1', 's321_1_1', 's542_1_2', 's432_1_1', 's531_1_2', 's362_1_2', 's342_2_1', 's311_1_1', 's351_1_1'],
//   ['s111_1_1', 's112_2_1', 's351_1_2', 's211_1_1', 's541_2_1', 's341_1_2', 's532_1_1', 's442_1_1', 's521_1_1', 's431_1_1', 's332_2_1', 's362_1_1'],
//   ['s111_1_1', 's112_2_1', 's531_1_2', 's341_2_1', 's361_1_2', 's432_1_1', 's521_1_1', 's311_1_1', 's352_1_2', 's332_1_1', 's442_1_2', 's541_1_1'],
//   ['s111_1_1', 's112_2_1', 's541_1_2', 's341_1_1', 's221_1_1', 's211_1_1', 's361_1_1', 's432_1_2', 's351_1_1', 's442_1_1', 's331_1_1', 's532_1_1'],
//   ['s111_1_1', 's112_2_1', 's352_1_1', 's362_1_1', 's441_1_1', 's532_1_2', 's431_1_1', 's341_1_1', 's331_2_1', 's542_2_1', 's321_1_2', 's411_1_2'],
//   ['s111_1_1', 's112_2_1', 's442_1_1', 's542_1_1', 's342_1_1', 's352_1_2', 's421_1_1', 's331_1_2', 's431_1_2', 's531_1_1', 's411_1_1', 's362_1_2'],
//   ['s111_1_1', 's112_2_1', 's332_1_1', 's511_1_1', 's542_1_2', 's361_1_1', 's532_1_2', 's421_1_2', 's432_1_1', 's351_1_1', 's342_1_2', 's441_1_2']
// ]

const sequences = [
  ['s111_1_1', 's112_2_1', 's441_1_1', 's531_1_1', 's351_1_1', 's331_1_2', 's371_1_1', 's543_1_1', 's341_2_1', 's541_2_1', 's521_1_2', 's544_1_2'],
  ['s111_1_1', 's112_2_1', 's342_2_1', 's544_1_1', 's351_1_2', 's543_1_2', 's221_1_1', 's331_2_1', 's543_1_1', 's442_1_1', 's541_1_1', 's371_1_1'],
  ['s111_1_1', 's112_2_1', 's543_1_2', 's321_1_1', 's531_1_1', 's352_1_1', 's544_1_1', 's332_1_2', 's541_1_2', 's342_1_2', 's441_1_1', 's371_1_1'],
  ['s111_1_1', 's112_2_1', 's531_1_1', 's542_2_1', 's352_1_1', 's544_1_2', 's441_1_2', 's332_2_1', 's543_1_2', 's221_1_2', 's342_1_1', 's371_1_2'],
  ['s111_1_1', 's112_2_1', 's543_1_1', 's371_1_1', 's352_1_1', 's341_1_2', 's542_1_1', 's332_1_2', 's371_1_1', 's441_1_1', 's544_1_1', 's544_1_1'],
  ['s111_1_1', 's112_2_1', 's442_1_2', 's331_1_1', 's321_1_1', 's542_1_2', 's544_1_1', 's531_1_2', 's371_1_2', 's342_2_1', 's543_1_1', 's351_1_1'],
  ['s111_1_1', 's112_2_1', 's351_1_2', 's543_1_1', 's541_2_1', 's341_1_2', 's371_1_1', 's442_1_1', 's521_1_1', 's544_1_1', 's332_2_1', 's371_1_1'],
  ['s111_1_1', 's112_2_1', 's531_1_2', 's341_2_1', 's371_1_2', 's544_1_1', 's521_1_1', 's543_1_1', 's352_1_2', 's332_1_1', 's442_1_2', 's541_1_1'],
  ['s111_1_1', 's112_2_1', 's541_1_2', 's341_1_1', 's221_1_1', 's543_1_1', 's371_1_1', 's544_1_2', 's351_1_1', 's442_1_1', 's331_1_1', 's543_1_1'],
  ['s111_1_1', 's112_2_1', 's352_1_1', 's371_1_1', 's441_1_1', 's221_1_1', 's544_1_1', 's341_1_1', 's331_2_1', 's542_2_1', 's321_1_2', 's543_1_2'],
  ['s111_1_1', 's112_2_1', 's442_1_1', 's542_1_1', 's342_1_1', 's352_1_2', 's544_1_1', 's331_1_2', 's544_1_2', 's531_1_1', 's543_1_1', 's371_1_2'],
  ['s111_1_1', 's112_2_1', 's332_1_1', 's543_1_1', 's542_1_2', 's521_1_1', 's371_1_2', 's342_1_2', 's544_1_1', 's351_1_1', 's342_1_2', 's441_1_2']
]



const initRef = useRef(false);

  useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;

    const initializeGame = async () => {
      const { gameSessionId, sessionNumber } = await createGameSession();
      setGameSessionId(gameSessionId);
      setSessionNumber(sessionNumber);
    };
    initializeGame();
  }, []);

  useEffect(() => {
    if (sessionNumber !== null) {
      const sequenceIndex = (sessionNumber - 1) % sequences.length;
      setCurrentSequence(sequences[sequenceIndex]);
    }
  }, [sessionNumber]);

  useEffect(() => {
    if (currentSequence.length > 0 && currentLevelIndex < currentSequence.length) {
      const [level, experienceType, pathNumber] = currentSequence[currentLevelIndex].split('_');
      const levelNumber = level.substring(1);
      
      const isExperienced = experienceType === '1';
      const experienceLevel = isExperienced ? 'experienced' : 'novice';
      const pathType = `${experienceLevel}${pathNumber}`;
      
      setCurrentLevelInfo({
        level: `level${levelNumber}`,
        pathType: pathType
      });
    }
  }, [currentSequence, currentLevelIndex]);

  const handleIntroductionComplete = useCallback(() => {
    setGameState('comprehensionCheck');
  }, []);

  const handleComprehensionCheckComplete = useCallback(async (actionLog) => {
    try {
      await updateGameSession(gameSessionId, { 
        comprehensionCheck: { actionLog, passed: true } 
      });
    } catch (error) {
      console.error("Error updating game session:", error);
    }
    setCurrentLevelIndex(0);
    setGameState('game');
  }, [gameSessionId]);

  const handleComprehensionCheckFailure = useCallback(async (actionLog) => {
    await updateGameSession(gameSessionId, { 
      comprehensionCheck: { actionLog, passed: false } 
    });
    alert('Some answers are incorrect. Please review the instructions and try again.');
    setGameState('introduction');
  }, [gameSessionId]);

  const handleLevelComplete = useCallback(async (won, remainingSteps) => {
    const currentLevel = currentLevelInfo.level;
    const levelOutcome = won ? 'completed' : 'timed out';
    console.log(`Level ${currentLevel} ${levelOutcome}`);
    
    try {
      await addGameLog(gameSessionId, {
        type: 'LEVEL_COMPLETE',
        data: { level: currentLevel, outcome: levelOutcome, remainingSteps }
      });
    } catch (error) {
      console.error("Error adding game log:", error);
    }
  
    if (currentLevelIndex === 1) {
      setTutorialCompleted(true);
      setGameState('tutorialComplete');
    } else if (currentLevelIndex >= 2) {
      const pointsEarned = remainingSteps;
      setLastTrialPoints(pointsEarned);
      setTotalPoints(prevTotal => prevTotal + pointsEarned);
      setGameState('trialComplete');
    }
    
    if (currentLevelIndex < currentSequence.length - 1) {
      setCurrentLevelIndex(prevIndex => prevIndex + 1);
    } else {
      setGameState('userInfo');
    }
  }, [currentLevelIndex, currentSequence, gameSessionId, currentLevelInfo]);

  const handleUserInfoSubmit = useCallback(async (info) => {
    try {
      await updateGameSession(gameSessionId, { 
        userData: info,
        completedAt: new Date().toISOString(),
        totalPoints: totalPoints  // Add this line to include total points
      });
  
      // Log the total points as a separate event
      await addGameLog(gameSessionId, {
        type: 'EXPERIMENT_COMPLETE',
        data: {
          totalPoints: totalPoints,
          completedAt: new Date().toISOString()
        }
      });
  
      setGameState('thankYou');
    } catch (error) {
      console.error("Error submitting user info and total points:", error);
      // You might want to show an error message to the user here
    }
  }, [gameSessionId, totalPoints]);  // Add totalPoints to the dependency array

  const handleDebugToggle = useCallback(() => {
    if (debugMode) {
      setDebugMode(false);
    } else {
      setShowPasswordModal(true);
    }
  }, [debugMode]);

  const handlePasswordSubmit = useCallback((password) => {
    if (password === 'cool123') {
      setDebugMode(true);
      setShowPasswordModal(false);
    } else {
      alert('Incorrect password');
    }
  }, []);

  const handleRetryTutorial = useCallback(() => {
    setCurrentLevelIndex(0);
    setTutorialCompleted(false);
    setGameState('game');
  }, []);

  const handleProceedToMainExperiment = useCallback(() => {
    setCurrentLevelIndex(2);
    setGameState('game');
  }, []);

  const getTutorialTrialNumber = (index) => `${index + 1}/2`;
  const getMainTrialNumber = (index) => `${index - 1}/${currentSequence.length - 2}`;

  return (
    <div className="App">
      {gameState === 'introduction' && (
        <GameIntroduction onComplete={handleIntroductionComplete} />
      )}
      {gameState === 'comprehensionCheck' && (
        <ComprehensionCheck 
          onComplete={handleComprehensionCheckComplete} 
          onFailure={handleComprehensionCheckFailure}
          gameSessionId={gameSessionId}
        />
      )}
      {gameState === 'game' && currentLevelInfo && !debugMode && (
        <GridGame 
          key={currentLevelIndex}
          onComplete={handleLevelComplete} 
          currentLevel={currentLevelInfo.level}
          pathType={currentLevelInfo.pathType}
          gameSessionId={gameSessionId}
          trialNumber={
            currentLevelIndex < 2 
              ? getTutorialTrialNumber(currentLevelIndex)
              : getMainTrialNumber(currentLevelIndex)
          }
          isTutorial={currentLevelIndex < 2}
        />
      )}
      {gameState === 'tutorialComplete' && (
        <div className="max-w-md mx-auto mt-8 p-6 bg-white rounded-lg shadow-md text-center">
          <h2 className="text-2xl font-bold mb-4">Tutorial Completed!</h2>
          <p className="mb-4">You have finished the tutorial. Click the buttons below to retry the two levels or proceed to the main experiment.</p>
          <div className="flex justify-center space-x-4">
            <button
              onClick={handleRetryTutorial}
              className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
            >
              Retry Tutorial
            </button>
            <button
              onClick={handleProceedToMainExperiment}
              className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded"
            >
              Proceed to Main Experiment
            </button>
          </div>
        </div>
      )}
      {gameState === 'trialComplete' && (
        <div className="max-w-md mx-auto mt-8 p-6 bg-white rounded-lg shadow-md text-center">
          <h2 className="text-2xl font-bold mb-4">Trial Completed!</h2>
          <p className="mb-4">You earned {lastTrialPoints} points this trial.</p>
          <p className="mb-4">Your total points: {totalPoints}</p>
          <button
            onClick={() => setGameState('game')}
            className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded"
          >
            Next Trial
          </button>
        </div>
      )}
      {gameState === 'userInfo' && (
        <UserInfoForm 
          onSubmit={handleUserInfoSubmit}
          gameSessionId={gameSessionId}
        />
      )}
      {gameState === 'thankYou' && (
        <div className="max-w-md mx-auto mt-8 p-6 bg-white rounded-lg shadow-md text-center">
          <h2 className="text-2xl font-bold mb-4">Thank You!</h2>
          <p>Your participation is greatly appreciated.</p> 
          <p></p>
          <p>Completion code: C6DFNH2C</p>
          <p className="mt-4">Total points earned: {totalPoints}</p>
        </div>
      )}
      {debugMode && (
        <div className="fixed top-0 right-0 m-4 space-y-2">
          <button onClick={() => setGameState('introduction')} className="block w-full bg-blue-500 text-white px-4 py-2 rounded">
            Instructions
          </button>
          <button onClick={() => setGameState('comprehensionCheck')} className="block w-full bg-green-500 text-white px-4 py-2 rounded">
            Comp Check
          </button>
          <button onClick={() => setGameState('game')} className="block w-full bg-yellow-500 text-white px-4 py-2 rounded">
            Test Mode
          </button>
          <button onClick={() => setGameState('userInfo')} className="block w-full bg-purple-500 text-white px-4 py-2 rounded">
            Info Form
          </button>
          <button onClick={() => setGameState('pathVisualization')} className="block w-full bg-pink-500 text-white px-4 py-2 rounded">
            Path Visualization
          </button>
        </div>
      )}
      {debugMode && gameState === 'pathVisualization' && (
        <GamePathVisualization />
      )}
      {debugMode && gameState === 'game' && (
        <TestModeGame />
      )}
      <button
        onClick={handleDebugToggle}
        className="fixed bottom-4 right-4 px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors"
      >
        {debugMode ? 'Disable Debug Mode' : 'Enable Debug Mode'}
      </button>
      {showPasswordModal && (
        <DebugPasswordModal onSubmit={handlePasswordSubmit} onClose={() => setShowPasswordModal(false)} />
      )}
    </div>
  );
};

export default App;