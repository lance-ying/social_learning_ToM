"use client"
import React, { useState } from 'react';
import { threeChestIntroductionText } from './introductions/threeChestIntroduction';
import { singleChestIntroductionText } from './introductions/singleChestIntroduction';
import { multiAgentIntroductionText } from './introductions/multi_exp3';
import { exp4IntroductionText } from './introductions/exp4_instructions';
import { EXPERIMENT_TYPE } from './introductions/config';
import DebugPasswordModal from '../debug/DebugPasswordModal';

interface GameIntroductionProps {
  onComplete: () => void;
  onDebugMode?: () => void;
}

const GameIntroduction: React.FC<GameIntroductionProps> = ({ onComplete, onDebugMode }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [showDebugModal, setShowDebugModal] = useState(false);
  const [videoCompleted, setVideoCompleted] = useState(false);
  
  // Select introduction text based on experiment type
  const introductionText = EXPERIMENT_TYPE === 'exp4'
    ? exp4IntroductionText
    : EXPERIMENT_TYPE === 'multi'
    ? multiAgentIntroductionText
    : EXPERIMENT_TYPE === 'single'
    ? singleChestIntroductionText
    : threeChestIntroductionText;

  // Debug log to verify which variant is being used
  console.log('GameIntroduction - EXPERIMENT_TYPE:', EXPERIMENT_TYPE);

  const handleNext = () => {
    if (currentStep < introductionText.length - 1) {
      setCurrentStep(currentStep + 1);
      setVideoCompleted(false); // Reset video completion for next step
    } else {
      onComplete();
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
      setVideoCompleted(false); // Reset video completion when going back
    }
  };

  const handleDebugPassword = (password: string) => {
    if (password === 'debug123') {
      setShowDebugModal(false);
      if (onDebugMode) {
        onDebugMode();
      }
    } else {
      alert('Incorrect password');
    }
  };

  const isLastStep = currentStep === introductionText.length - 1;
  // Steps 5-9 are the video/image demo pages for exp4 (0-indexed) - hide right panel
  const isMediaStep = EXPERIMENT_TYPE === 'exp4' && (currentStep >= 5 && currentStep <= 9);
  // Only steps 5, 6, 7, 8 have videos that need completion checking (step 9 is an image)
  const isVideoStep = EXPERIMENT_TYPE === 'exp4' && (currentStep >= 5 && currentStep <= 8);
  const shouldShowRightPanel = !isLastStep && !isMediaStep;

  // Add handlers to video elements and completion text recursively
  const addVideoHandler = (element: any): any => {
    if (!React.isValidElement(element)) {
      return element;
    }

    if (element.type === 'video') {
      return React.cloneElement(element as any, {
        onLoadedMetadata: (e: React.SyntheticEvent<HTMLVideoElement>) => {
          e.currentTarget.playbackRate = 1.5;
        },
        onTimeUpdate: (e: React.SyntheticEvent<HTMLVideoElement>) => {
          const video = e.currentTarget;
          // Mark as completed when video reaches 95% or more
          if (video.currentTime / video.duration >= 0.95 && !videoCompleted) {
            setVideoCompleted(true);
          }
        },
      });
    }

    const elementProps = element.props as any;
    // Handle video-completion-text class
    if (elementProps?.className && typeof elementProps.className === 'string' &&
        elementProps.className.includes('video-completion-text')) {
      return React.cloneElement(element as any, {
        className: elementProps.className.replace('opacity-0', videoCompleted ? 'opacity-100' : 'opacity-0')
      });
    }

    if (elementProps && elementProps.children) {
      return React.cloneElement(element as any, {
        children: React.Children.map(elementProps.children, addVideoHandler)
      });
    }

    return element;
  };

  const renderContent = () => {
    const content = introductionText[currentStep];
    if (isVideoStep && React.isValidElement(content)) {
      return addVideoHandler(content);
    }
    return content;
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100 p-4">
      {/* Debug Mode Button - positioned at top right */}
      {onDebugMode && (
        <button
          onClick={() => setShowDebugModal(true)}
          className="fixed top-4 right-4 px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors z-10"
        >
          Debug Mode
        </button>
      )}
      
      <div className={`max-w-6xl w-full mx-auto bg-white rounded-lg shadow-md ${isLastStep || isMediaStep ? 'p-6' : 'flex flex-col md:flex-row'}`}>
        <div className={`flex-1 p-6 flex flex-col justify-between ${isLastStep || isMediaStep ? 'w-full' : ''}`}>
          <div className={`whitespace-pre-line mb-6 text-black ${isLastStep ? 'text-lg' : ''}`}>
            {renderContent()}
          </div>
          <div className="flex justify-between">
            <button
              onClick={handleBack}
              className={`bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-2 px-4 rounded ${currentStep === 0 ? 'invisible' : ''}`}
            >
              ◄ Back
            </button>
            <button
              onClick={handleNext}
              disabled={isVideoStep && !videoCompleted}
              className={`font-bold py-2 px-4 rounded ${
                isVideoStep && !videoCompleted
                  ? 'bg-gray-400 text-gray-600 cursor-not-allowed'
                  : 'bg-green-500 hover:bg-green-600 text-white'
              }`}
            >
              {currentStep < introductionText.length - 1 ? 'Next ►' : 'Start Game'}
            </button>
          </div>
        </div>
        {shouldShowRightPanel && (
          <div className="flex-1 p-6 flex flex-col">
            <img
              src="./icons/game_screen_new.png"
              alt="Game Screenshot"
              className="w-full h-auto rounded-lg shadow-md mb-4"
            />
          </div>
        )}
      </div>

      {showDebugModal && (
        <DebugPasswordModal
          onSubmit={handleDebugPassword}
          onClose={() => setShowDebugModal(false)}
        />
      )}
    </div>
  );
};

export default GameIntroduction;