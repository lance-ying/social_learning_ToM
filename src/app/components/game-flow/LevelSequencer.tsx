"use client"
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { firebaseLogger } from '@/services/FirebaseLogger';
import { getDisplayScore, getDisplayedPointsForLevel } from '@/utils/scoreDisplay';

// Level sequences from the old App.js
// const sequences = [
//   ['mod_s111_1', 'mod_s112_2', 'mod_s531_1', 'mod_s342_2', 'mod_s411_1', 'mod_s362_1', 'mod_s342_1', 'mod_s442_2', 'mod_s432_2', 'mod_s541_2', 'mod_s361_2', 'mod_s511_1'],
//   ['mod_s111_2', 'mod_s112_1', 'mod_s532_2', 'mod_s431_2', 'mod_s211_2', 'mod_s341_2', 'mod_s331_2', 'mod_s521_1', 'mod_s351_2', 'mod_s544_1', 'mod_s311_1', 'mod_s221_1'],
//   ['mod_s111_1', 'mod_s112_2', 'mod_s542_2', 'mod_s531_2', 'mod_s362_2', 'mod_s211_1', 'mod_s421_1', 'mod_s432_1', 'mod_s371_1', 'mod_s542_1', 'mod_s341_1', 'mod_s442_1'],
//   ['mod_s111_2', 'mod_s112_1', 'mod_s371_2', 'mod_s431_1', 'mod_s352_2', 'mod_s361_1', 'mod_s543_2', 'mod_s521_2', 'mod_s332_1', 'mod_s532_1', 'mod_s441_2', 'mod_s411_2'],
//   ['mod_s111_1', 'mod_s112_2', 'mod_s331_1', 'mod_s541_1', 'mod_s511_2', 'mod_s321_1', 'mod_s544_2', 'mod_s441_1', 'mod_s351_1', 'mod_s332_2', 'mod_s321_2', 'mod_s311_2'],
//   ['mod_s111_2', 'mod_s112_1', 'mod_s543_1', 'mod_s221_2', 'mod_s352_1', 'mod_s311_1', 'mod_s352_2', 'mod_s421_2', 'mod_s531_1', 'mod_s342_1', 'mod_s543_2', 'mod_s411_1'],
//   ['mod_s111_1', 'mod_s112_2', 'mod_s211_1', 'mod_s362_1', 'mod_s351_2', 'mod_s532_1', 'mod_s432_1', 'mod_s521_2', 'mod_s371_1', 'mod_s542_1', 'mod_s432_2', 'mod_s521_1'],
//   ['mod_s111_2', 'mod_s112_1', 'mod_s421_1', 'mod_s371_2', 'mod_s331_2', 'mod_s361_2', 'mod_s442_2', 'mod_s532_2', 'mod_s544_1', 'mod_s431_1', 'mod_s341_2', 'mod_s342_2'],
//   ['mod_s111_1', 'mod_s112_2', 'mod_s361_1', 'mod_s442_1', 'mod_s511_1', 'mod_s362_2', 'mod_s531_2', 'mod_s542_2', 'mod_s341_1', 'mod_s211_2', 'mod_s431_2', 'mod_s221_1'],
//   ['mod_s111_2', 'mod_s112_1', 'mod_s541_2', 'mod_s332_1', 'mod_s331_1', 'mod_s332_2', 'mod_s421_2', 'mod_s411_2', 'mod_s352_1', 'mod_s544_2', 'mod_s221_2', 'mod_s511_2'],
//   ['mod_s111_1', 'mod_s112_2', 'mod_s321_2', 'mod_s543_1', 'mod_s441_2', 'mod_s311_2', 'mod_s441_1', 'mod_s541_1', 'mod_s321_1', 'mod_s351_1', 'mod_s531_1', 'mod_s541_2'],
//   ['mod_s111_2', 'mod_s112_1', 'mod_s543_2', 'mod_s361_1', 'mod_s352_2', 'mod_s521_2', 'mod_s342_2', 'mod_s211_1', 'mod_s361_2', 'mod_s542_1', 'mod_s542_2', 'mod_s341_1'],
//   ['mod_s111_1', 'mod_s112_2', 'mod_s341_2', 'mod_s432_2', 'mod_s331_2', 'mod_s421_1', 'mod_s332_1', 'mod_s532_2', 'mod_s371_1', 'mod_s544_1', 'mod_s311_1', 'mod_s362_2'],
//   ['mod_s111_2', 'mod_s112_1', 'mod_s442_2', 'mod_s531_2', 'mod_s432_1', 'mod_s511_1', 'mod_s351_2', 'mod_s362_1', 'mod_s442_1', 'mod_s221_1', 'mod_s342_1', 'mod_s211_2'],
//   ['mod_s111_1', 'mod_s112_2', 'mod_s532_1', 'mod_s411_1', 'mod_s371_2', 'mod_s521_1', 'mod_s431_1', 'mod_s431_2', 'mod_s331_1', 'mod_s321_2', 'mod_s541_1', 'mod_s352_1'],
//   ['mod_s111_2', 'mod_s112_1', 'mod_s511_2', 'mod_s351_1', 'mod_s441_2', 'mod_s321_1', 'mod_s311_2', 'mod_s543_1', 'mod_s332_2', 'mod_s421_2', 'mod_s441_1', 'mod_s342_1'],
//   ['mod_s111_1', 'mod_s112_2', 'mod_s544_2', 'mod_s221_2', 'mod_s411_2', 'mod_s371_1', 'mod_s371_2', 'mod_s532_1', 'mod_s311_1', 'mod_s341_1', 'mod_s362_1', 'mod_s332_1'],
//   ['mod_s111_2', 'mod_s112_1', 'mod_s521_1', 'mod_s541_2', 'mod_s431_1', 'mod_s543_2', 'mod_s532_2', 'mod_s442_1', 'mod_s341_2', 'mod_s211_1', 'mod_s331_2', 'mod_s351_2'],
//   ['mod_s111_1', 'mod_s112_2', 'mod_s362_2', 'mod_s531_1', 'mod_s431_2', 'mod_s521_2', 'mod_s531_2', 'mod_s544_1', 'mod_s542_1', 'mod_s352_2', 'mod_s411_1', 'mod_s361_1'],
//   ['mod_s111_2', 'mod_s112_1', 'mod_s211_2', 'mod_s342_2', 'mod_s432_2', 'mod_s361_2', 'mod_s432_1', 'mod_s221_1', 'mod_s442_2', 'mod_s511_1', 'mod_s542_2', 'mod_s421_1'],
// ];

// const sequences = [
//   ['mod_s111_1', 'mod_s112_1', 'mod_s221_1', 'mod_s352_1', 'mod_s442_1', 'mod_s311_1', 'mod_s531_1', 'mod_s543_1', 'mod_s341_1', 'mod_s421_1', 'mod_s432_1', 'mod_s411_1'],
//   ['mod_s111_1', 'mod_s112_1', 'mod_s342_1', 'mod_s211_1', 'mod_s532_1', 'mod_s371_1', 'mod_s541_1', 'mod_s361_1', 'mod_s441_1', 'mod_s362_1', 'mod_s521_1', 'mod_s332_1'],
//   ['mod_s111_1', 'mod_s112_1', 'mod_s351_1', 'mod_s542_1', 'mod_s331_1', 'mod_s511_1', 'mod_s321_1', 'mod_s544_1', 'mod_s431_1', 'mod_s441_1', 'mod_s332_1', 'mod_s341_1'],
//   ['mod_s111_1', 'mod_s112_1', 'mod_s321_1', 'mod_s331_1', 'mod_s351_1', 'mod_s532_1', 'mod_s543_1', 'mod_s432_1', 'mod_s411_1', 'mod_s421_1', 'mod_s361_1', 'mod_s542_1'],
//   ['mod_s111_1', 'mod_s112_1', 'mod_s544_1', 'mod_s511_1', 'mod_s362_1', 'mod_s541_1', 'mod_s521_1', 'mod_s442_1', 'mod_s531_1', 'mod_s311_1', 'mod_s342_1', 'mod_s221_1'],
//   ['mod_s111_1', 'mod_s112_1', 'mod_s371_1', 'mod_s211_1', 'mod_s352_1', 'mod_s431_1', 'mod_s532_1', 'mod_s362_1', 'mod_s211_1', 'mod_s543_1', 'mod_s371_1', 'mod_s421_1'],
//   ['mod_s111_1', 'mod_s112_1', 'mod_s332_1', 'mod_s541_1', 'mod_s411_1', 'mod_s521_1', 'mod_s341_1', 'mod_s432_1', 'mod_s342_1', 'mod_s531_1', 'mod_s352_1', 'mod_s442_1'],
//   ['mod_s111_1', 'mod_s112_1', 'mod_s544_1', 'mod_s351_1', 'mod_s311_1', 'mod_s441_1', 'mod_s221_1', 'mod_s361_1', 'mod_s511_1', 'mod_s542_1', 'mod_s331_1', 'mod_s321_1'],
// ];

// const sequences = [
//   ['mod_s111_1', 'mod_s112_1', 'mod_s351_1', 'mod_s432_1', 'mod_s542_1', 'mod_s521_1', 'mod_s211_1', 'mod_s544_1', 'mod_s371_1', 'mod_s331_1', 'mod_s311_1', 'mod_s332_1'],
//   ['mod_s111_1', 'mod_s112_1', 'mod_s442_1', 'mod_s541_1', 'mod_s511_1', 'mod_s362_1', 'mod_s321_1', 'mod_s532_1', 'mod_s543_1', 'mod_s361_1', 'mod_s371_1', 'mod_s311_1'],
//   ['mod_s111_1', 'mod_s112_1', 'mod_s441_1', 'mod_s531_1', 'mod_s331_1', 'mod_s542_1', 'mod_s511_1', 'mod_s361_1', 'mod_s543_1', 'mod_s321_1', 'mod_s351_1', 'mod_s441_1'],
//   ['mod_s111_1', 'mod_s112_1', 'mod_s521_1', 'mod_s442_1', 'mod_s532_1', 'mod_s332_1', 'mod_s211_1', 'mod_s544_1', 'mod_s541_1', 'mod_s432_1', 'mod_s362_1', 'mod_s321_1'],
//   ['mod_s111_1', 'mod_s112_1', 'mod_s531_1', 'mod_s442_1', 'mod_s362_1', 'mod_s531_1', 'mod_s311_1', 'mod_s351_1', 'mod_s331_1', 'mod_s542_1', 'mod_s432_1', 'mod_s543_1'],
//   ['mod_s111_1', 'mod_s112_1', 'mod_s544_1', 'mod_s441_1', 'mod_s511_1', 'mod_s532_1', 'mod_s361_1', 'mod_s521_1', 'mod_s371_1', 'mod_s211_1', 'mod_s541_1', 'mod_s332_1'],
//   ['mod_s111_1', 'mod_s112_1', 'mod_s371_1', 'mod_s432_1', 'mod_s442_1', 'mod_s521_1', 'mod_s531_1', 'mod_s351_1', 'mod_s544_1', 'mod_s321_1', 'mod_s362_1', 'mod_s331_1'],
//   ['mod_s111_1', 'mod_s112_1', 'mod_s541_1', 'mod_s511_1', 'mod_s361_1', 'mod_s532_1', 'mod_s543_1', 'mod_s441_1', 'mod_s211_1', 'mod_s332_1', 'mod_s321_1', 'mod_s331_1'],
//   ['mod_s111_1', 'mod_s112_1', 'mod_s311_1', 'mod_s542_1', 'mod_s361_1', 'mod_s441_1', 'mod_s211_1', 'mod_s543_1', 'mod_s542_1', 'mod_s521_1', 'mod_s332_1', 'mod_s532_1'],
//   ['mod_s111_1', 'mod_s112_1', 'mod_s371_1', 'mod_s511_1', 'mod_s541_1', 'mod_s442_1', 'mod_s362_1', 'mod_s544_1', 'mod_s351_1', 'mod_s311_1', 'mod_s432_1', 'mod_s531_1'],
// ];

// const sequences = [
//   // ['sm111_1', 'sm112_1', 'sm321_2', 'sm331_1', 'sm543_2', 'sm431_1', 'sm521_2', 'sm411_2', 'sm342_2', 'sm541_2', 'sm361_2', 'sm351_2'],
//   // ['sm111_1', 'sm112_1', 'sm332_2', 'sm432_1', 'sm541_1', 'sm321_1', 'sm311_1', 'sm371_1', 'sm341_1', 'sm531_1', 'sm543_1', 'sm421_1'],
//   ['sm111_1', 'sm112_1', 'sm332_1', 'sm511_2', 'sm341_2', 'sm371_2', 'sm361_1', 'sm331_2', 'sm311_2', 'sm421_2', 'sm431_2', 'sm351_1'],
//   // ['sm111_1', 'sm112_1', 'sm411_1', 'sm531_2', 'sm342_1', 'sm432_2', 'sm511_1', 'sm521_1', 'sm332_1', 'sm541_1', 'sm361_2', 'sm331_2'],
//   // ['sm111_1', 'sm112_1', 'sm543_2', 'sm341_1', 'sm521_2', 'sm371_1', 'sm431_1', 'sm332_2', 'sm311_2', 'sm321_1', 'sm351_2', 'sm541_2'],
//   // ['sm111_1', 'sm112_1', 'sm311_1', 'sm421_1', 'sm432_1', 'sm341_2', 'sm411_2', 'sm321_2', 'sm361_1', 'sm342_2', 'sm331_1', 'sm543_1'],
//   ['sm111_1', 'sm112_1', 'sm371_2', 'sm531_1', 'sm421_2', 'sm511_2', 'sm431_2', 'sm342_1', 'sm521_1', 'sm351_1', 'sm411_1', 'sm332_2'],
//   ['sm111_1', 'sm112_1', 'sm432_2', 'sm531_2', 'sm511_1', 'sm321_1', 'sm371_1', 'sm332_1', 'sm351_2', 'sm521_2', 'sm543_1', 'sm311_2'],
//   // ['sm111_1', 'sm112_1', 'sm331_1', 'sm541_1', 'sm543_2', 'sm421_2', 'sm431_1', 'sm341_1', 'sm311_1', 'sm321_2', 'sm371_2', 'sm361_2'],
//   // ['sm111_1', 'sm112_1', 'sm531_1', 'sm341_2', 'sm361_1', 'sm511_2', 'sm541_2', 'sm331_2', 'sm432_1', 'sm421_1', 'sm342_2', 'sm411_2'],
// ];

// const sequences = [
//   ['sm111_1', 'sm112_1', 'sm211_1', 'sm221_1', 'sm431_1', 'sm611_1', 'sm543_1'],
//   ['sm111_1', 'sm112_1', 'sm221_2', 'sm211_2', 'sm543_2', 'sm431_2', 'sm611_2'],
//   ['sm111_1', 'sm112_1', 'sm431_1', 'sm543_1', 'sm211_1', 'sm221_2', 'sm611_1'],
//   ['sm111_1', 'sm112_1', 'sm611_2', 'sm431_2', 'sm221_1', 'sm543_2', 'sm211_2'],
//   ['sm111_1', 'sm112_1', 'sm543_1', 'sm611_1', 'sm431_2', 'sm211_1', 'sm221_1'],
//   ['sm111_1', 'sm112_1', 'sm211_2', 'sm221_2', 'sm611_2', 'sm543_1', 'sm431_1'],
//   ['sm111_1', 'sm112_1', 'sm221_1', 'sm431_1', 'sm543_2', 'sm611_2', 'sm211_1'],
//   ['sm111_1', 'sm112_1', 'sm611_1', 'sm211_2', 'sm221_2', 'sm431_2', 'sm543_2'],
//   ['sm111_1', 'sm112_1', 'sm431_2', 'sm543_1', 'sm611_1', 'sm221_1', 'sm211_2'],
//   ['sm111_1', 'sm112_1', 'sm211_1', 'sm611_2', 'sm221_2', 'sm543_2', 'sm431_1'],
// ];
//
const sequences = [
  ['sm111_1', 'sm112_1', 'sm531_2', 'sm432_1', 'sm543_1', 'sm331_1', 'sm341_2', 'sm541_1', 'sm321_1', 'sm612_1', 'sm411_1', 'sm371_1'],
  ['sm111_1', 'sm112_1', 'sm421_1', 'sm531_1', 'sm351_2', 'sm612_2', 'sm371_2', 'sm411_2', 'sm431_1', 'sm543_2', 'sm331_2', 'sm341_1'],
  ['sm111_1', 'sm112_1', 'sm311_1', 'sm611_1', 'sm421_2', 'sm541_2', 'sm211_2', 'sm431_2', 'sm321_2', 'sm361_2', 'sm351_1', 'sm531_1'],
  ['sm111_1', 'sm112_1', 'sm211_1', 'sm432_2', 'sm521_2', 'sm611_2', 'sm511_1', 'sm361_1', 'sm221_2', 'sm341_1', 'sm371_1', 'sm331_2'],
  ['sm111_1', 'sm112_1', 'sm311_2', 'sm521_1', 'sm531_2', 'sm371_2', 'sm511_2', 'sm221_1', 'sm431_1', 'sm541_1', 'sm611_2', 'sm543_2'],
  ['sm111_1', 'sm112_1', 'sm331_1', 'sm421_1', 'sm612_2', 'sm341_2', 'sm411_2', 'sm543_1', 'sm351_2', 'sm432_1', 'sm321_1', 'sm211_1'],
  ['sm111_1', 'sm112_1', 'sm612_1', 'sm411_1', 'sm541_2', 'sm361_2', 'sm521_1', 'sm432_2', 'sm511_2', 'sm371_1', 'sm221_2', 'sm341_1'],
  ['sm111_1', 'sm112_1', 'sm511_1', 'sm421_2', 'sm611_1', 'sm361_1', 'sm351_1', 'sm431_2', 'sm211_2', 'sm521_2', 'sm543_1', 'sm341_2'],
  ['sm111_1', 'sm112_1', 'sm321_2', 'sm311_2', 'sm331_1', 'sm431_1', 'sm221_1', 'sm543_2', 'sm612_2', 'sm531_2', 'sm411_2', 'sm371_2'],
  ['sm111_1', 'sm112_1', 'sm311_1', 'sm331_2', 'sm612_1', 'sm541_1', 'sm351_2', 'sm321_1', 'sm531_1', 'sm432_1', 'sm411_1', 'sm421_1'],
];

// const sequences = [
//   ['s111_1', 's112_2', 's341_2', 's331_1', 's511_2', 's543_1', 's541_1', 's532_1', 's221_1', 's442_2', 's351_2', 's431_1'],
//   ['s111_2', 's112_1', 's441_2', 's361_2', 's542_1', 's543_2', 's371_2', 's342_2', 's332_1', 's531_1', 's521_2', 's511_1'],
//   ['s111_1', 's112_2', 's332_2', 's532_2', 's432_2', 's541_2', 's521_1', 's331_2', 's544_2', 's311_2', 's421_2', 's341_1'],
//   ['s111_2', 's112_1', 's411_2', 's342_1', 's361_1', 's431_2', 's531_2', 's351_1', 's321_1', 's544_1', 's371_1', 's441_1'],
//   ['s111_1', 's112_2', 's421_1', 's311_1', 's542_2', 's432_1', 's442_1', 's411_1', 's221_2', 's543_1', 's331_1', 's341_2'],
//   ['s111_2', 's112_1', 's321_2', 's511_2', 's442_2', 's531_1', 's371_1', 's332_1', 's351_2', 's544_2', 's221_2', 's431_1'],
//   ['s111_1', 's112_2', 's311_2', 's432_2', 's341_1', 's321_1', 's542_2', 's361_2', 's532_1', 's544_1', 's411_2', 's421_2'],
//   ['s111_2', 's112_1', 's442_1', 's431_2', 's221_1', 's371_2', 's541_1', 's543_2', 's532_2', 's321_2', 's342_1', 's351_1'],
//   ['s111_1', 's112_2', 's432_1', 's342_2', 's311_1', 's421_1', 's331_2', 's541_2', 's511_1', 's521_2', 's441_2', 's332_2'],
//   ['s111_2', 's112_1', 's521_1', 's542_1', 's441_1', 's411_1', 's531_2', 's361_1', 's543_1', 's511_2', 's331_1', 's341_2'],
// ];

interface LevelInfo {
  level: string;
  pathType: string;
}

interface LevelSequencerProps {
  gameSessionId: string;
  sessionNumber: number;
  onLevelChange: (levelInfo: LevelInfo, levelIndex: number, isTutorial: boolean) => void;
  onTutorialComplete: () => void;
  onTrialComplete: (pointsEarned: number, displayedPointsEarned: number) => void;
  onExperimentComplete: () => void;
}

export interface LevelCompleteInfo {
  won: boolean;
  remainingSteps: number;
}

const useLevelSequencer = ({
  gameSessionId,
  sessionNumber,
  onLevelChange,
  onTutorialComplete,
  onTrialComplete,
  onExperimentComplete
}: LevelSequencerProps) => {
  const [currentLevelIndex, setCurrentLevelIndex] = useState(0);
  const [currentSequence, setCurrentSequence] = useState<string[]>([]);
  const [currentLevelInfo, setCurrentLevelInfo] = useState<LevelInfo | null>(null);
  const [tutorialCompleted, setTutorialCompleted] = useState(false);
  const [totalPoints, setTotalPoints] = useState(0);
  const [totalDisplayedPoints, setTotalDisplayedPoints] = useState(0);

  // Initialize sequence based on session number
  useEffect(() => {
    if (sessionNumber !== null) {
      const sequenceIndex = (sessionNumber - 1) % sequences.length;
      setCurrentSequence(sequences[sequenceIndex]);
    }
  }, [sessionNumber]);

  // Parse level info when sequence or level index changes
  useEffect(() => {
    if (currentSequence.length > 0 && currentLevelIndex < currentSequence.length) {
      // Old code (doesn't work with mod_ prefix):
      // const [levelId, pathNumber] = currentSequence[currentLevelIndex].split('_');
      // // Entry is like 's211_1' or 's211_2'
      // // levelId = 's211', pathNumber = '1' or '2'

      // New code (works with mod_sXXX_1 format):
      const fullLevelId = currentSequence[currentLevelIndex];
      // Entry is like 'mod_s211_1' or 's211_1'
      // Extract the path number (last part after final underscore)
      const lastUnderscoreIndex = fullLevelId.lastIndexOf('_');
      const levelId = fullLevelId.substring(0, lastUnderscoreIndex); // 'mod_s211' or 's211'
      const pathNumber = fullLevelId.substring(lastUnderscoreIndex + 1); // '1' or '2'

      const levelInfo: LevelInfo = {
        level: levelId,  // Full level ID: mod_s211, s211, etc.
        pathType: `experienced${pathNumber}`  // Map 1 -> experienced1, 2 -> experienced2
      };

      setCurrentLevelInfo(levelInfo);

      // Notify parent about level change
      const isTutorial = currentLevelIndex < 2;
      onLevelChange(levelInfo, currentLevelIndex, isTutorial);
    }
  }, [currentSequence, currentLevelIndex, onLevelChange]);

  const handleLevelComplete = useCallback(async (levelCompleteInfo: LevelCompleteInfo) => {
    const { won, remainingSteps } = levelCompleteInfo;
    const currentLevel = currentLevelInfo?.level;
    const currentPathType = currentLevelInfo?.pathType;

    // Get the full level identifier with variant (e.g., 's211_1' or 's211_2')
    const fullLevelId = currentSequence[currentLevelIndex] || currentLevel;
    const levelOutcome = won ? 'completed' : 'timed out';

    console.log(`Level ${fullLevelId} (${currentLevel} - ${currentPathType}) ${levelOutcome}`);

    // Calculate displayed score (clamped to 0 minimum)
    const displayedRemainingSteps = getDisplayScore(remainingSteps);

    // Calculate displayed points earned (0 for tutorials, clamped for main trials)
    const displayedPointsEarned = getDisplayedPointsForLevel(remainingSteps, currentLevelIndex);

    try {
      // Log to Firebase if enabled
      if (firebaseLogger.isLoggingEnabled()) {
        await firebaseLogger.logEvent('level_complete', {
          level: currentLevel,
          levelWithVariant: fullLevelId,  // Include full ID like 's211_1'
          pathType: currentPathType,       // Include path type like 'experienced1'
          outcome: levelOutcome,
          remainingSteps: remainingSteps,  // Actual score (can be negative)
          displayedRemainingSteps: displayedRemainingSteps,  // Displayed score (min 0)
          levelIndex: currentLevelIndex,
          isTutorial: currentLevelIndex < 2
        }, fullLevelId || 'unknown');
      }
    } catch (error) {
      console.error("Error logging level complete:", error);
    }

    // Handle tutorial completion (after level index 1, which is the 2nd level)
    if (currentLevelIndex === 1) {
      setTutorialCompleted(true);
      onTutorialComplete();
    }
    // Handle main experiment trials (level index 2+)
    else if (currentLevelIndex >= 2) {
      const pointsEarned = remainingSteps;  // Actual points (can be negative)
      setTotalPoints(prevTotal => prevTotal + pointsEarned);
      setTotalDisplayedPoints(prevTotal => prevTotal + displayedPointsEarned);
      onTrialComplete(pointsEarned, displayedPointsEarned);
    }

    // Progress to next level or complete experiment
    if (currentLevelIndex < currentSequence.length - 1) {
      setCurrentLevelIndex(prevIndex => prevIndex + 1);
    } else {
      // All levels completed
      onExperimentComplete();
    }
  }, [currentLevelIndex, currentSequence, currentLevelInfo, onTutorialComplete, onTrialComplete, onExperimentComplete]);

  const retryTutorial = useCallback(() => {
    setCurrentLevelIndex(0);
    setTutorialCompleted(false);
  }, []);

  const proceedToMainExperiment = useCallback(() => {
    setCurrentLevelIndex(2);
  }, []);

  const getTutorialTrialNumber = (index: number) => `${index + 1}/2`;
  const getMainTrialNumber = (index: number) => `${index - 1}/${currentSequence.length - 2}`;

  return {
    currentLevelInfo,
    currentLevelIndex,
    currentSequence,
    tutorialCompleted,
    totalPoints,
    totalDisplayedPoints,
    handleLevelComplete,
    retryTutorial,
    proceedToMainExperiment,
    getTutorialTrialNumber,
    getMainTrialNumber,
    isTutorial: currentLevelIndex < 2,
    isMainExperiment: currentLevelIndex >= 2
  };
};

export default useLevelSequencer;
