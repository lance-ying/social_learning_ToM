import React from 'react';
import Image from 'next/image';
import { PiTreasureChestBold } from "react-icons/pi";
import { GiNestedHexagons, GiGemPendant } from "react-icons/gi";

interface IconProps {
  name: 'treasure' | 'barrier' | 'amulet' | 'player' | 'otherPlayerBlue' | 'otherPlayerGreen' | 'redWizard' | 'blueWizard';
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
      return <Image src="/icons/bl.png" alt="Player" width={20} height={24} className="inline" />;
    case 'otherPlayerBlue':
      return <Image src="/icons/al.png" alt="Blue Player" width={16} height={20} className="inline" />;
    case 'otherPlayerGreen':
      return <Image src="/icons/green_a.png" alt="Green Player" width={16} height={20} className="inline" />;
    case 'redWizard':
      return <Image src="/icons/f.png" alt="Red Wizard" width={24} height={24} className="inline" />;
    case 'blueWizard':
      return <Image src="/icons/d.png" alt="Blue Wizard" width={24} height={24} className="inline" />;
    default:
      return null;
  }
};

export const multiAgentIntroductionText = [
  `Welcome to our treasure hunting game!\n\nBefore you begin your task, you'll complete a brief guided tutorial (~ 3 minutes) to understand the game.\n\nPress Next to continue.`,

  <>
    {`You are playing a treasure game with three possible treasure pots: A, B, and C. `}<Icon name="treasure" color="text-yellow-500" />{` Your goal is to get one of the treasure pots. In each trial, we will specify which treasure you should get.\n\nEach treasure pot is hidden behind a barrier `}<Icon name="barrier" color="text-blue-500" />{`. You need to possess certain amulets to pass the barrier. There are two kinds of amulets: red `}<Icon name="amulet" color="text-red-500" />{` and blue `}<Icon name="amulet" color="text-blue-500" />
  </>,

  <>
    {`Players can get the amulets from wizards. There is only 1 red and 1 blue amulet in each trial. There are two kinds of wizards in the game:\n\n• The red wizard `}<Icon name="redWizard" />{` possesses the red amulet\n• The blue wizards `}<Icon name="blueWizard" />{` have either the blue amulet or have nothing to give. There are exactly 1 red wizard and multiple blue wizards in each trial. Among the blue wizards, only 1 blue wizard has the blue amulet.\n\nYou can interact with characters and objects through movements. For example, if you are to the left of a wizard/treasure/barrier and you perform the "right" action, you will interact with it.`}
  </>,

  <>
    {`There are TWO other players who are also playing the game:\n\n• `}<Icon name="otherPlayerBlue" />{` Player 2 (blue)\n• `}<Icon name="otherPlayerGreen" />{` Player 3 (green)\n\nPlayer 2 and Player 3 are each pursuing one of the three goals, which may or may not be the same as yours.\n\nIn each timestep, you can choose to `}<strong>observe one of the other players</strong>{` or `}<strong>perform an action yourself</strong>{`. The actions each cost different points.\n\n• Observe Player 2 (blue): `}<strong>1 point</strong>{` \n• Observe Player 3 (green): `}<strong>1 point</strong>{` \n• Movement: `}<strong>3 points</strong>{` \n• Interacting with a wizard: `}<strong>5 points</strong>{` \n\nThe other players only move when you choose to observe them. If you choose to observe a player, you will stay still for one step and watch that player's action. If you choose to perform an action, both other players will stay still while you act.\n\nThe colored trails behind each agent indicate their respective paths (blue for `}<Icon name="otherPlayerBlue" />{` Player 2 and green for `}<Icon name="otherPlayerGreen" />{` Player 3).\n\nBoth Player 2 and Player 3 are expert players with full knowledge of which wizard has the amulet. They will always follow the most efficient path. The other players' goals will be given to you in each trial.`}
  </>,


<div className="flex flex-col items-center space-y-4">
  <Image src="/icons/game_multi_3.gif" alt="Gameplay demonstration" width={640} height={480} className="w-full max-w-2xl rounded-lg border-2 border-gray-300" />
  <div className="text-lg space-y-2">
    <p><strong>Important:</strong> Pay attention to <strong className="text-blue-600">notification box</strong> at the top of the screen!</p>
    <p>When you observe a player, you will see messages showing what they obtain from wizards. Player 2 is represented by the <Icon name="otherPlayerBlue" /> icon and Player 3 is represented by the <Icon name="otherPlayerGreen" /> icon in notifications.</p>
  </div>
</div>,

  <>
    {`You will start with a set amount of points at each trial. Your goal is to reach the treasure with the most amount of points remaining, which you will earn at the end of each trial. At the end of the experiment, you will receive a bonus for the total points you earned`} <strong>(50 points = 0.5 dollar, capped at 1 dollar)</strong> {`.\n
    Please note that:\n\n• Players can walk past each other\n• Each wizard has 1 or no amulet\n• Treasure pots have enough treasures for each player\n• You need your own amulets to pass barriers\n• You cannot see the other players' inventory\n• You can only observe but not interact with the other players\n• You can observe either Player 2 or Player 3 each turn\n\n`}<strong>Hint:</strong>{` Observing Player 2 or Player 3 can help you infer the location of amulets.\n\n`}<strong>Hint:</strong>{` Pay attention to the location of Player 2 and Player 3 and think about which goals they are going for.\n\nLet's go through 2 trial runs!`}
  </>,

<div className="text-lg">
Please note that this is a study on how and when do people learn from others.
<br /><br />
Your task is to find the treasure by <strong>observing Player 2 or Player 3 when you think it will be helpful.</strong>
<br /><br />
This can be done by clicking either <strong>"Observe Player 2"</strong> or <strong>"Observe Player 3"</strong> button.
<br /><br />
<div className="text-xl">
  <strong>Please pay attention to which treasure goal (A, B, or C) you need to focus on in each trial, and think carefully about when observing the other players would be useful for completing your goal efficiently.</strong>
  <br /><br />
  It is ok to not observe in some trials if you think you can find the treasure on your own.
  <br /><br />
  We will reject submissions if you show low effort on the task.
</div>
</div>
];

export const multiAgentComprehensionQuestions = [
  {
    id: 'task',
    question: 'What is your task in this experiment?',
    options: [
      { key: 'option1', text: 'Reach treasure chest in map' },
      { key: 'option2', text: 'Prevent other players from getting the treasure' },
      { key: 'option3', text: 'Help other players collect treasure' }
    ],
    correctAnswer: 'Reach treasure chest in map'
  },
  {
    id: 'barrier',
    question: 'If treasure is behind a barrier that requires a red amulet, what should you do?',
    options: [
      { key: 'option1', text: 'Get red amulet from the red wizard' },
      { key: 'option2', text: 'Get red amulet from one of the blue wizards' },
      { key: 'option3', text: 'Try to break through the barrier directly' }
    ],
    correctAnswer: 'Get red amulet from the red wizard'
  },
  {
    id: 'players',
    question: 'What happens to other players when you perform your own action (instead of observing)?',
    options: [
      { key: 'option1', text: 'Both blue and green players stay still' },
      { key: 'option2', text: 'The blue player moves but the green player stays still' },
      { key: 'option3', text: 'Both players continue moving as normal' }
    ],
    correctAnswer: 'Both blue and green players stay still'
  },
  {
    id: 'observe',
    question: 'When you observe a player, what happens?',
    options: [
      { key: 'option1', text: 'Both you and observed player move' },
      { key: 'option2', text: 'You stay still and watch the observed player move' },
      { key: 'option3', text: 'The observed player stays still while you move' }
    ],
    correctAnswer: 'You stay still and watch the observed player move'
  }
];