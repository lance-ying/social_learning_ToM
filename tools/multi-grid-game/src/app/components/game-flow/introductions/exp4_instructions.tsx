import React from 'react';
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
      return <img src="/icons/bl.png" alt="Player" className="inline w-5 h-6" />;
    case 'otherPlayerBlue':
      return <img src="/icons/al.png" alt="Blue Player" className="inline w-4 h-5" />;
    case 'otherPlayerGreen':
      return <img src="/icons/green_a.png" alt="Green Player" className="inline w-4 h-5" />;
    case 'redWizard':
      return <img src="./icons/f.png" alt="Red Wizard" className="inline w-6 h-6" />;
    case 'blueWizard':
      return <img src="./icons/d.png" alt="Blue Wizard" className="inline w-6 h-6" />;
    default:
      return null;
  }
};

export const exp4IntroductionText = [
  `Welcome to our treasure hunting game!\n\nBefore you begin your task, you'll complete a brief guided tutorial (~ 3 minutes) to understand the game.\n\nPress Next to continue.`,

  <>
    {`You are playing a treasure game with three possible treasure pots: A, B, and C. `}<Icon name="treasure" color="text-yellow-500" />{` Your goal is to get one of the treasure pots. In each trial, we will specify which treasure you should get.\n\nEach treasure pot is hidden behind a barrier `}<Icon name="barrier" color="text-blue-500" />{`. You need to possess certain amulets to pass the barrier. There are two kinds of amulets: red `}<Icon name="amulet" color="text-red-500" />{` and blue `}<Icon name="amulet" color="text-blue-500" />
  </>,

  <>
    {`Players can get the amulets from wizards. There is only 1 red and 1 blue amulet in each trial. There are two kinds of wizards in the game:\n\n• The red wizard `}<Icon name="redWizard" />{` possesses the red amulet\n• The blue wizards `}<Icon name="blueWizard" />{` have either the blue amulet or have nothing to give. There are exactly 1 red wizard and multiple blue wizards in each trial. Among the blue wizards, only 1 blue wizard has the blue amulet.\n\nYou can interact with characters and objects through movements. For example, if you are to the left of a wizard/treasure/barrier and you perform the "right" action, you will interact with it.`}
  </>,

  <>
    {`There are TWO other players who are also playing the game:\n\n• `}<Icon name="otherPlayerBlue" />{` Player 2 (blue)\n• `}<Icon name="otherPlayerGreen" />{` Player 3 (green)\n\nEach player is pursuing one of the three goals, which may or may not be the same as yours.\n\nIn each timestep, you can choose to `}<strong>observe one of the other players</strong>{` or `}<strong>perform an action yourself</strong>{`. The actions each cost different points.\n\n• Observe Player 2 (blue): `}<strong>1 point</strong>{`\n• Observe Player 3 (green): `}<strong>1 point</strong>{`\n• Movement: `}<strong>3 points</strong>{`\n• Interacting with a wizard: `}<strong>5 points</strong>{`\n\nThe other players only move when you choose to observe them. If you choose to observe a player, you will stay still for one step and watch that player's action. If you choose to perform an action, the other players will stay still while you act.\n\nThe colored trail behind each agent indicates their path (blue for `}<Icon name="otherPlayerBlue" />{` Player 2, green for `}<Icon name="otherPlayerGreen" />{` Player 3).`}
  </>,

  <>
    <div className="text-lg space-y-4">
      <p><strong>A key detail to pay attention to is the experience level of each agent.</strong></p>

      <p><strong className="text-orange-600">Novice agents</strong> have the same experience level as you, meaning that they don't know which of the blue wizards has the blue amulet. The novice agent CAN see the map as you do, but only doesn't have hidden knowledge about where the blue amulet is among the blue wizards.</p>

      <p><strong className="text-purple-600">Expert agents</strong>, on the other hand, do have full knowledge of where the blue amulet is located.</p>

      <p>For example, if a <strong className="text-orange-600">Novice</strong> agent's goal is to reach a treasure chest blocked by a blue barrier, the Novice agent will move toward and interact with the <em>closest</em> blue wizard, and continue doing this loop <strong>UNTIL</strong> the agent has interacted with the wizard that actually has the blue amulet. The moment the novice obtains the blue amulet, the novice will take the optimal path toward their treasure.</p>

      <p>An <strong className="text-purple-600">Expert</strong> agent, however, will go directly to the correct blue wizard and take the most efficient path.</p>

      <p><strong>Let's watch 2 examples:</strong></p>
    </div>
  </>,

  <div className="flex flex-col items-center space-y-4">
    <p className="text-lg"><strong>Example 1:</strong> Both Player 2 and Player 3 are <strong className="text-purple-600">Experts</strong></p>
    <div className="flex gap-6 w-full items-start">
      <div className="flex-1 video-completion-text opacity-0 transition-opacity duration-500 text-lg space-y-2">
        <p>In this example, both Player 2 and Player 3 are <strong className="text-purple-600">Experts</strong>. Therefore, they always perform optimally and know where the blue amulet is hidden among the blue wizards.</p>
        <p>Notice how both agents take direct paths to the correct wizard.</p>
      </div>
      <div className="flex-1">
        <video
          src="/icons/demo_1_0.mov"
          className="w-full rounded-lg border-2 border-gray-300"
          autoPlay
          muted
          loop
          playsInline
        />
      </div>
    </div>
  </div>,

  <div className="flex flex-col items-center space-y-4">
    <p className="text-xl"><strong>Example 2:</strong> Player 2 is an <strong className="text-purple-600">Expert</strong></p>
    <video
      src="/icons/demo_2_1.mov"
      className="w-full max-w-2xl rounded-lg border-2 border-gray-300"
      autoPlay
      muted
      loop
      playsInline
    />
  </div>,

  <div className="flex flex-col items-center space-y-4">
    <p className="text-xl"><strong>Example 2 (continued):</strong> Player 3 is a <strong className="text-orange-600">Novice</strong></p>
    <video
      src="/icons/demo_2_2.mov"
      className="w-full max-w-2xl rounded-lg border-2 border-gray-300"
      autoPlay
      muted
      loop
      playsInline
    />
  </div>,

  <div className="flex flex-col items-center space-y-6">
    <p className="text-xl"><strong>Example 2 Comparison:</strong> <strong className="text-purple-600">Expert</strong> vs <strong className="text-orange-600">Novice</strong></p>
    <div className="flex gap-6 w-full items-start">
      <div className="flex-1 space-y-4">
        <div className="text-2xl space-y-4">
          <p>In this example, Player 2 is an <strong className="text-purple-600">Expert</strong> while Player 3 is a <strong className="text-orange-600">Novice</strong>.</p>
          <p><strong>Pay attention to the pathing:</strong> Both agents are attempting to reach their goal, but Player 3 (the Novice) interacts with multiple blue wizards because the first wizard they tried didn't have the amulet.</p>
        </div>
      </div>
      <div className="flex-1 flex gap-4">
        <div className="flex-1">
          <video
            src="/icons/demo_2_2.mov"
            className="w-full rounded-lg border-2 border-gray-300"
            autoPlay
            muted
            loop
            playsInline
          />
          <p className="text-center text-lg mt-2"><strong className="text-purple-600">Expert Path (Player 2)</strong></p>
        </div>
        <div className="flex-1">
          <video
            src="/icons/demo_2_1.mov"
            className="w-full rounded-lg border-2 border-gray-300"
            autoPlay
            muted
            loop
            playsInline
          />
          <p className="text-center text-lg mt-2"><strong className="text-orange-600">Novice Path (Player 3)</strong></p>
        </div>
      </div>
    </div>
  </div>,

  <div className="flex flex-col items-center space-y-4">
    <p className="text-xl"><strong>Let's take a closer look at the pathing from the novice Player 3:</strong></p>
    <img
      src="/icons/demo_2_img.png"
      alt="Novice Player Pathing Analysis"
      className="w-full max-w-2xl rounded-lg border-2 border-gray-300"
    />
    <div className="text-xl space-y-4">
      <p><strong className="text-blue-600">IMPORTANT:</strong> After the novice Player 3 interacts with the initial wizard, the player moves toward the other wizard. <strong>Given that specific movement cue, you can then understand that the first wizard did NOT have a blue amulet to give.</strong></p>
      <p>It is important to make sure that you are observing carefully and meaningfully. Failure to do so may result in inefficient gameplay and lost points.</p>
    </div>
  </div>,

  <div className="text-lg space-y-4">
      <p><strong>Important:</strong> Pay attention to the <strong className="text-blue-600">notification box</strong> at the top of the screen!</p>
      <p>When you observe a player, you will see a message showing they interacted with a wizard (but not which item they received). Player 2 is represented by the <Icon name="otherPlayerBlue" /> icon and Player 3 is represented by the <Icon name="otherPlayerGreen" /> icon in notifications.</p>
      <p>Additionally, <strong>badges appear on top of each agent</strong> to indicate their status: your character is marked as the <strong>main player</strong>, while Player 2 and Player 3 have badges showing whether they are <strong className="text-purple-600">Experts</strong> or <strong className="text-orange-600">Novices</strong>. Pay attention to these badges so you know the experience level of each agent you are observing.</p>
    </div>,

  <>
    {`You will start with a set amount of points at each trial. Your goal is to reach the treasure with the most amount of points remaining, which you will earn at the end of each trial. At the end of the experiment, you will receive a bonus for the total points you earned`} <strong>(60 points = 0.5 cents, capped at 1 dollar)</strong> {`.\n
    Please note that:\n\n• Players can walk past each other\n• Each wizard has 1 or no amulet\n• Treasure pots have enough treasures for each player\n• You need your own amulets to pass barriers\n• You cannot see the other players' inventories\n• You can only observe but not interact with the other players\n\n`}<strong>Hint:</strong>{` Observing players can help you infer the location of amulets.\n\nLet's go through 2 trial runs!`}
  </>,

  <div className="text-lg">
    Please note that this is a study on how and when do people learn from others.
    <br /><br />
    You will complete <strong>10 trials</strong> in total (after the 2 practice trials you just completed).
    <br /><br />
    Your task is to find the treasure by <strong>observing Player 2 or Player 3 when you think it will be helpful.</strong>
    <br /><br />
    This can be done by clicking the <strong>"Observe Player 2"</strong> or <strong>"Observe Player 3"</strong> button.
    <br /><br />
    <div className="text-xl">
      <strong>Please pay attention to which treasure goal (A, B, or C) you need to focus on in each trial, and think carefully about when observing each player would be useful for completing your goal efficiently.</strong>
      <br /><br />
      It is ok to not observe in some trials if you think you can find the treasure on your own.
      <br /><br />
      <strong className="text-red-600">We will reject submissions if you show low effort on the task or have very low scoring across the trials.</strong>
    </div>
  </div>
];

export const exp4ComprehensionQuestions = [
  {
    id: 'expert_vs_novice',
    question: 'What is the key difference between an Expert agent and a Novice agent?',
    options: [
      { key: 'option1', text: 'Expert agents know which blue wizard has the amulet, Novice agents do not' },
      { key: 'option2', text: 'Expert agents move faster than Novice agents' },
      { key: 'option3', text: 'Expert agents can break through barriers without amulets' }
    ],
    correctAnswer: 'Expert agents know which blue wizard has the amulet, Novice agents do not'
  },
  {
    id: 'novice_behavior',
    question: 'If a Novice agent needs a blue amulet to reach their treasure, what will they do?',
    options: [
      { key: 'option1', text: 'Go directly to the blue wizard that has the amulet' },
      { key: 'option2', text: 'Check the closest blue wizard and repeat this loop until obtaining the blue amulet' },
      { key: 'option3', text: 'Skip getting the amulet and break through the barrier' }
    ],
    correctAnswer: 'Check the closest blue wizard and repeat this loop until obtaining the blue amulet'
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
    id: 'players',
    question: 'What happens to the other players when you perform your own action (instead of observing)?',
    options: [
      { key: 'option1', text: 'The other players stay still' },
      { key: 'option2', text: 'The other players continue moving as normal' },
      { key: 'option3', text: 'The other players move twice as fast' }
    ],
    correctAnswer: 'The other players stay still'
  },
  {
    id: 'observe',
    question: 'When you observe a player, what do you see in the notifications?',
    options: [
      { key: 'option1', text: 'You see that the player interacted with a wizard (without details about what they received)' },
      { key: 'option2', text: 'You see exactly which amulet the player received' },
      { key: 'option3', text: 'You see nothing about the player\'s interactions' }
    ],
    correctAnswer: 'You see that the player interacted with a wizard (without details about what they received)'
  },
  {
    id: 'novice_inference',
    question: 'If a Novice agent interacts with a wizard and then moves toward another wizard, what does that tell you?',
    options: [
      { key: 'option1', text: 'The first wizard had the amulet they needed' },
      { key: 'option2', text: 'The first wizard did NOT have the amulet they needed' },
      { key: 'option3', text: 'The agent is confused and doesn\'t know what they are doing' }
    ],
    correctAnswer: 'The first wizard did NOT have the amulet they needed'
  },
  {
    id: 'who_to_observe',
    question: 'Who should you observe to help you find the treasure most efficiently?',
    options: [
      { key: 'option1', text: 'Always observe the Expert agent' },
      { key: 'option2', text: 'Always observe the Novice agent' },
      { key: 'option3', text: 'It depends on the map layout, positioning of wizards, and your goal' },
      { key: 'option4', text: 'Never observe - you should always act on your own' }
    ],
    correctAnswer: 'It depends on the map layout, positioning of wizards, and your goal'
  }
];
