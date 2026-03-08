import React, { useState } from 'react';
import { PiTreasureChestBold } from "react-icons/pi";
import { GiNestedHexagons, GiGemPendant } from "react-icons/gi";

const GameIntroduction = ({ onComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);

  const Icon = ({ name, color, size = 24 }) => {
    switch (name) {
      case 'treasure':
        return <PiTreasureChestBold size={size} className={`inline ${color}`} />;
      case 'barrier':
        return <GiNestedHexagons size={size} className={`inline ${color}`} />;
      case 'amulet':
        return <GiGemPendant size={size} className={`inline ${color}`} />;
      case 'player':
        return <img src="/icons/bl.png" alt="Player" className="inline w-5 h-6" />;
      case 'otherPlayer':
        return <img src="/icons/al.png" alt="Other Player" className="inline w-4 h-5" />;
      case 'redWizard':
        return <img src="./icons/f.png" alt="Red Wizard" className="inline w-6 h-6" />;
      case 'blueWizard':
        return <img src="./icons/d.png" alt="Blue Wizard" className="inline w-6 h-6" />;
      default:
        return null;
    }
  };

  const introductionText = [
    `Welcome to our treasure hunting game!\n\nBefore you begin your task, you'll complete a brief guided tutorial (~ 3 minutes) to understand the game.\n\nPress Next to continue.`,

    <>
      {`You are playing a treasure game with three possible treasure pots: A, B, and C. `}<Icon name="treasure" color="text-yellow-500" />{` Your goal is to get one of the treasure pots. In each trial, we will specify which treasure you should get.\n\nEach treasure pot is hidden behind a barrier `}<Icon name="barrier" color="text-blue-500" />{`. You need to possess certain amulets to pass the barrier. There are two kinds of amulets: red `}<Icon name="amulet" color="text-red-500" />{` and blue `}<Icon name="amulet" color="text-blue-500" />
    </>,

    <>
      {`Players can get the amulets from wizards. There is only 1 red and 1 blue amulet in each trial. There are two kinds of wizards in the game:\n\n• The red wizard `}<Icon name="redWizard" />{` possesses the red amulet\n• The blue wizards `}<Icon name="blueWizard" />{` have either the blue amulet or have nothing to give. There are exactly 1 red wizard and multiple blue wizards in each trial. Among the blue wizards, only 1 blue wizard has the blue amulet.\n\nYou can interact with characters and objects through movements. For example, if you are to the left of a wizard/treasure/barrier and you perform the "right" action, you will interact with it.`}
    </>,

    <>
      {`There is another player `}<Icon name="otherPlayer" />{` who is also playing the game. They are pursuing one of the three goals, which may or may not be the same as yours.\n\nIn each timestep, you can choose to `}<strong>observe the other player</strong>{` or `}<strong>perform an action yourself</strong>{`. The actions each cost different points.\n\n• Observe: `}<strong>1 point</strong>{` \n• Movement: `}<strong>2 point</strong>{` \n• Interacting with a wizard: `}<strong>5 points</strong>{` \n
      If you choose to observe, you will stay still for one step, while the other agent will perform an action. Note, the other agent will only move when you choose to observe. \n\n The blue trail behind the other agent indicates their path.\n\n You can only observe a maximum of 25 steps in each trial.\n
      You may be paired with one of the two kinds of players: `}<strong>advanced</strong>{` and `}<strong>expert</strong>{`. The expert player has full knowledge of which wizard has the amulet and they will always follow the most efficient path, whereas the advanced player may occasionally make mistake (i.e. they know which blue wizard may have the amulet). The other player's level will be given to you in each trial.
      `}
    </>,

    <>
      {`You will start with a set amount of points at each trial. Your goal is to reach the treasure with the most amount of points remaining, which you will earn at the end of each trial. At the end of the experiment, you will receive a bonus for the total points you earned`} <strong>(100 points = 0.5 dollar, capped at 1 dollar)</strong> {`.\n
      Please note that:\n\n• Players can walk past each other\n• Each wizard has 1 or no amulet\n• Treasure pots have enough treasures for each player\n• You need your own amulets to pass barriers\n• You cannot see the other player's inventory\n• You can only observe but not interact with the other player\n\n`}<strong>Hint:</strong>{` Observing the other player can help you infer the location of amulets.\n\n`}<strong>Hint:</strong>{` If you realize that the other player is farther away from the wizards than you or they are not heading towards the similar goal, you should stop observing them.\n\nLet's go through 2 trial runs!`}
    </>,

    <div className="text-lg">
      Please note that this is a study on how and when do people learn from others.
      <br /><br />
      Your task is to find the treasure by <strong>observing the other player when necessary.</strong>
      <br /><br />
      This can be done by clicking the <strong>"Observe"</strong> option.
      <br /><br />
      Your data would not be useful for us if you never observe the other player in all 10 trials and we would <strong>reject your submission</strong> in these cases.
      It is ok to not observe in some trials if you think the other player is less helpful.
    </div>
  ];

  const handleNext = () => {
    if (currentStep < introductionText.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onComplete();
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const isLastStep = currentStep === introductionText.length - 1;

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className={`max-w-6xl w-full mx-auto bg-white rounded-lg shadow-md ${isLastStep ? 'p-6' : 'flex flex-col md:flex-row'}`}>
        <div className={`flex-1 p-6 flex flex-col justify-between ${isLastStep ? 'w-full' : ''}`}>
          <div className={`whitespace-pre-line mb-6 ${isLastStep ? 'text-lg' : ''}`}>
            {introductionText[currentStep]}
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
              className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded"
            >
              {currentStep < introductionText.length - 1 ? 'Next ►' : 'Start Game'}
            </button>
          </div>
        </div>
        {!isLastStep && (
          <div className="flex-1 p-6 flex flex-col">
            <img 
              src="./icons/game_screen.png" 
              alt="Game Screenshot" 
              className="w-full h-auto rounded-lg shadow-md mb-4"
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default GameIntroduction;