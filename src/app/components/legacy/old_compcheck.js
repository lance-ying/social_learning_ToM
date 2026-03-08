import React, { useState, useCallback, useEffect } from 'react';
import { addComprehensionCheck, addGameLog, getComprehensionCheck } from './firestoreHelpers';

const ComprehensionCheck = ({ onComplete, onFailure, gameSessionId }) => {
  const [answers, setAnswers] = useState({
    task: '',
    barrier: '',
    goal: ''
  });
  const [actionLog, setActionLog] = useState([]);
  const [failedAttempts, setFailedAttempts] = useState(() => {
    const storedAttempts = localStorage.getItem(`failedAttempts_${gameSessionId}`);
    return storedAttempts ? parseInt(storedAttempts, 10) : 0;
  });

  useEffect(() => {
    const fetchFailedAttempts = async () => {
      try {
        const checkData = await getComprehensionCheck(gameSessionId);
        if (checkData && checkData.failedAttempts !== undefined) {
          const storedAttempts = Math.max(checkData.failedAttempts, failedAttempts);
          setFailedAttempts(storedAttempts);
          localStorage.setItem(`failedAttempts_${gameSessionId}`, storedAttempts.toString());
        }
      } catch (error) {
        console.error("Error fetching failed attempts:", error);
      }
    };

    fetchFailedAttempts();
  }, [gameSessionId, failedAttempts]);

  const questions = [
    {
      id: 'task',
      question: 'What is the task of this experiment?',
      options: [
        'Collect one of the treasure boxes',
        'Collect all the treasure boxes',
        'Watch and help the other player collect treasure boxes'
      ],
      correctAnswer: 'Collect one of the treasure boxes'
    },
    {
      id: 'barrier',
      question: 'The treasure box is hidden behind a red door, what should you do?',
      options: [
        'Find a red amulet from the red wizard',
        'Find a red amulet from one of the blue wizards',
        'Try to go through the barrier directly'
      ],
      correctAnswer: 'Find a red amulet from the red wizard'
    },
    {
      id: 'goal',
      question: 'What is the goal of the other player?',
      options: [
        'Collect the same treasure box than yours',
        'Collect a different treasure box than yours',
        'Collect one of the boxes, which may be the same or different from yours'
      ],
      correctAnswer: 'Collect one of the boxes, which may be the same or different from yours'
    }
  ];

  const handleAnswerChange = useCallback((questionId, answer) => {
    setAnswers(prev => ({ ...prev, [questionId]: answer }));
    setActionLog(prevLog => [...prevLog, {
      timestamp: new Date().toISOString(),
      action: 'select_answer',
      questionId,
      selectedAnswer: answer
    }]);
  }, []);

  const handleSubmit = useCallback(async () => {
    const allCorrect = questions.every(q => answers[q.id] === q.correctAnswer);
    
    const newFailedAttempts = allCorrect ? failedAttempts : failedAttempts + 1;
    setFailedAttempts(newFailedAttempts);
    localStorage.setItem(`failedAttempts_${gameSessionId}`, newFailedAttempts.toString());
    
    const finalActionLog = [
      ...actionLog,
      {
        timestamp: new Date().toISOString(),
        action: 'submit_answers',
        finalAnswers: answers,
        failedAttempts: newFailedAttempts
      }
    ];

    const checkData = {
      answers,
      passed: allCorrect,
      actionLog: finalActionLog,
      failedAttempts: newFailedAttempts
    };

    try {
      await addComprehensionCheck(gameSessionId, checkData);
      await addGameLog(gameSessionId, {
        type: 'COMPREHENSION_CHECK',
        result: allCorrect ? 'passed' : 'failed',
        failedAttempts: newFailedAttempts
      });

      if (allCorrect) {
        onComplete(finalActionLog);
      } else {
        onFailure(finalActionLog);
      }
    } catch (error) {
      console.error("Error submitting comprehension check:", error);
    }
  }, [answers, questions, onComplete, onFailure, gameSessionId, actionLog, failedAttempts]);

  return (
    <div className="max-w-2xl mx-auto mt-8 p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-6">Comprehension Check</h2>
      <p className="mb-4">Failed attempts: {failedAttempts}</p>
      {questions.map(q => (
        <div key={q.id} className="mb-6">
          <p className="font-semibold mb-2">{q.question}</p>
          {q.options.map(option => (
            <div key={option} className="flex items-center mb-2">
              <input
                type="radio"
                id={`${q.id}-${option}`}
                name={q.id}
                value={option}
                checked={answers[q.id] === option}
                onChange={() => handleAnswerChange(q.id, option)}
                className="mr-2"
              />
              <label htmlFor={`${q.id}-${option}`}>{option}</label>
            </div>
          ))}
        </div>
      ))}
      <button
        onClick={handleSubmit}
        className="w-full bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600 transition-colors"
      >
        Submit
      </button>
    </div>
  );
};

export default ComprehensionCheck;