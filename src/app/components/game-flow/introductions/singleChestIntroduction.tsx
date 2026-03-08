import React from 'react';
import { PiTreasureChestBold } from "react-icons/pi";
import { GiNestedHexagons, GiGemPendant } from "react-icons/gi";

interface IconProps {
  name: 'treasure' | 'barrier' | 'amulet' | 'player' | 'otherPlayer' | 'redWizard' | 'blueWizard';
  color?: string;
  size?: number;
}

const Icon: React.FC<IconProps> = ({ name, color, size = 24 }) => {
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

export const singleChestIntroductionText = [
  `Welcome to our treasure hunting game!\n\nBefore you begin your task, you'll complete a brief guided tutorial (~ 3 minutes) to understand the game.\n\nPress Next to continue.`,

  <>
    {`You are playing a treasure game with one treasure chest hidden in the map. `}<Icon name="treasure" color="text-yellow-500" />{` Your goal is to get the treasure chest specified for the trial.\n\nThe treasure is hidden behind a barrier `}<Icon name="barrier" color="text-blue-500" />{`. You need to possess certain amulets to pass the barrier. There are two kinds of amulets: red `}<Icon name="amulet" color="text-red-500" />{` and blue `}<Icon name="amulet" color="text-blue-500" />
  </>,

  <>
    {`Players can get the amulets from wizards. There is only 1 red and 1 blue amulet in each trial. There are two kinds of wizards in the game:\n\n• The red wizard `}<Icon name="redWizard" />{` possesses the red amulet\n• The blue wizards `}<Icon name="blueWizard" />{` have either the blue amulet or have nothing to give. There are exactly 1 red wizard and multiple blue wizards in each trial. Among the blue wizards, only 1 blue wizard has the blue amulet.\n\nYou can interact with characters and objects through movements. For example, if you are to the left of a wizard/treasure/barrier and you perform the "right" action, you will interact with it.`}
  </>,

  <>
    {`There is another player `}<Icon name="otherPlayer" />{` who is also playing the game. They \n\nIn each timestep, you can choose to `}<strong>observe the other player</strong>{` or `}<strong>perform an action yourself</strong>{`. The actions each cost different points.\n\n• Observe: `}<strong>1 point</strong>{` \n• Movement: `}<strong>3 point</strong>{` \n• Interacting with a wizard: `}<strong>5 points</strong>{` \n
    The other player only moves when you choose to observe. If you choose to observe, you will stay still for one step and watch the other player's action. If you choose to perform an action, the other player will stay still while you act. \n\n The blue trail behind the other agent indicates their path.\n
    The other player is an expert player and has full knowledge of which wizard has the amulet. They will always follow the most efficient path.
    `}
  </>,


<div className="flex flex-col items-center space-y-4">
  <img src="/icons/gameplay_single.gif" alt="Gameplay demonstration" className="w-full max-w-2xl rounded-lg border-2 border-gray-300" />
  <div className="text-lg space-y-2">
    <p><strong>Important:</strong> Pay attention to the <strong className="text-blue-600">notification box</strong> at the top of the screen!</p>
    <p>When you observe the other player, you will see messages showing what they obtain from wizards. The other player is represented by their <Icon name="otherPlayer" /> icon in the notifications.</p>
  </div>
</div>,

  <>
    {`You will start with a set amount of points at each trial. Your goal is to reach the treasure with the most amount of points remaining, which you will earn at the end of each trial. At the end of the experiment, you will receive a bonus for the total points you earned`} <strong>(100 points = 0.5 dollar, capped at 1 dollar)</strong> {`.\n
    Please note that:\n\n• Players can walk past each other\n• Each wizard has 1 or no amulet\n• The treasure has enough for each player\n• You need your own amulets to pass barriers\n• You cannot see the other player's inventory\n• You can only observe but not interact with the other player\n\n`}<strong>Hint:</strong>{` Observing the other player can help you infer the location of amulets.\n\n`}<strong>Hint:</strong>{` Pay attention to the location of the other player and think about which wizard they are going for\n\nLet's go through 2 trial runs!`}
  </>,

<div className="text-lg">
Please note that this is a study on how and when do people learn from others.
<br /><br />
Your task is to find the treasure by <strong>observing the other player when necessary.</strong>
<br /><br />
This can be done by clicking the <strong>"Observe"</strong> option.
<br /><br />
<div className="text-xl">
  Your data would not be useful for us if you never observe the other player in all 10 trials and we would <strong>reject your submission</strong> in these cases.
  It is ok to not observe in some trials if you think the other player is less helpful.
  We will reject submissions if you show low effort on the task.
</div>
</div>
];

export const singleChestComprehensionQuestions = [
  {
    id: 'task',
    question: 'What is your task in this experiment?',
    options: [
      { key: 'option1', text: 'Reach the treasure chest in the map' },
      { key: 'option2', text: 'Prevent the other player from getting the treasure' },
      { key: 'option3', text: 'Help the other player collect the treasure' }
    ],
    correctAnswer: 'Reach the treasure chest in the map'
  },
  {
    id: 'barrier',
    question: 'If the treasure is behind a barrier that requires a red amulet, what should you do?',
    options: [
      { key: 'option1', text: 'Get the red amulet from the red wizard' },
      { key: 'option2', text: 'Get the red amulet from one of the blue wizards' },
      { key: 'option3', text: 'Try to break through the barrier directly' }
    ],
    correctAnswer: 'Get the red amulet from the red wizard'
  },
  {
    id: 'goal',
    question: 'Since there is only one treasure chest, what can you say about the other player?',
    options: [
      { key: 'option1', text: 'They are trying to reach the same treasure as you' },
      { key: 'option2', text: 'They are trying to help you get the treasure' },
      { key: 'option3', text: 'They have a different goal than you' }
    ],
    correctAnswer: 'They are trying to reach the same treasure as you'
  }
];