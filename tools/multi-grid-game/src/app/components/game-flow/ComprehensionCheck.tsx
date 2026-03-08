"use client";
import React, { useState, useCallback, useEffect } from "react";
import { firebaseLogger } from "@/services/FirebaseLogger";
import { threeChestComprehensionQuestions } from "./introductions/threeChestIntroduction";
import { singleChestComprehensionQuestions } from "./introductions/singleChestIntroduction";
import { multiAgentComprehensionQuestions } from "./introductions/multi_exp3";
import { exp4ComprehensionQuestions } from "./introductions/exp4_instructions";
import { EXPERIMENT_TYPE } from "./introductions/config";

interface ComprehensionCheckProps {
  onComplete: (actionLog: any[]) => void;
  onFailure: (actionLog: any[]) => void;
  onReturnToInstructions?: () => void;
  gameSessionId: string;
}

interface Question {
  id: string;
  question: string;
  options: { key: string; text: string; }[];
  correctAnswer: string;
}

// Changed to support dynamic question IDs
interface Answers {
  [key: string]: string;
}

interface ActionLogEntry {
  timestamp: string;
  action: string;
  questionId?: string;
  selectedAnswer?: string;
  finalAnswers?: Answers;
  failedAttempts?: number;
  feedbackByQuestion?: Record<string, "correct" | "incorrect">;
}

const ComprehensionCheck: React.FC<ComprehensionCheckProps> = ({
  onComplete,
  onFailure,
  onReturnToInstructions,
  gameSessionId,
}) => {
  const questions: Question[] = EXPERIMENT_TYPE === 'exp4'
    ? exp4ComprehensionQuestions
    : EXPERIMENT_TYPE === 'multi'
    ? multiAgentComprehensionQuestions
    : EXPERIMENT_TYPE === 'single'
    ? singleChestComprehensionQuestions
    : threeChestComprehensionQuestions;

  const [answers, setAnswers] = useState<Answers>({});
  const [actionLog, setActionLog] = useState<ActionLogEntry[]>([]);
  const [failedAttempts, setFailedAttempts] = useState(() => {
    const storedAttempts = localStorage.getItem(
      `failedAttempts_${gameSessionId}`
    );
    return storedAttempts ? parseInt(storedAttempts, 10) : 0;
  });

  // Feedback state
  const [feedbackByQuestion, setFeedbackByQuestion] = useState<
    Record<string, "correct" | "incorrect" | null>
  >({});
  const [hasSubmitted, setHasSubmitted] = useState(false);

  // Ensure answers/feedback objects include all current question IDs
  useEffect(() => {
    setAnswers((prev) => {
      const updated: Answers = { ...prev };
      questions.forEach((q) => {
        if (updated[q.id] === undefined) updated[q.id] = "";
      });
      return updated;
    });

    setFeedbackByQuestion((prev) => {
      const updated: Record<string, "correct" | "incorrect" | null> = {
        ...prev,
      };
      questions.forEach((q) => {
        if (updated[q.id] === undefined) updated[q.id] = null;
      });
      return updated;
    });
  }, [questions]);

  // Keep failed attempts in sync with localStorage
  useEffect(() => {
    try {
      const storedAttempts = localStorage.getItem(
        `failedAttempts_${gameSessionId}`
      );
      if (storedAttempts) {
        const attempts = parseInt(storedAttempts, 10);
        setFailedAttempts(attempts);
      }
    } catch (error) {
      console.error("Error fetching failed attempts:", error);
    }
  }, [gameSessionId]);

  const handleAnswerChange = useCallback((questionId: string, answer: string) => {
    setAnswers((prev) => ({ ...prev, [questionId]: answer }));
    setActionLog((prevLog) => [
      ...prevLog,
      {
        timestamp: new Date().toISOString(),
        action: "select_answer",
        questionId,
        selectedAnswer: answer,
      },
    ]);
  }, []);

  const handleSubmit = useCallback(async () => {
    // Build per-question correctness
    const perQuestion: Record<string, "correct" | "incorrect"> = {};
    questions.forEach((q) => {
      perQuestion[q.id] =
        answers[q.id] && answers[q.id] === q.correctAnswer
          ? "correct"
          : "incorrect";
    });

    const allCorrect = Object.values(perQuestion).every(
      (v) => v === "correct"
    );

    setFeedbackByQuestion((prev) => {
      const next = { ...prev };
      questions.forEach((q) => {
        next[q.id] = perQuestion[q.id];
      });
      return next;
    });
    setHasSubmitted(true);

    const newFailedAttempts = allCorrect ? failedAttempts : failedAttempts + 1;
    setFailedAttempts(newFailedAttempts);
    localStorage.setItem(
      `failedAttempts_${gameSessionId}`,
      newFailedAttempts.toString()
    );

    const finalActionLog: ActionLogEntry[] = [
      ...actionLog,
      {
        timestamp: new Date().toISOString(),
        action: "submit_answers",
        finalAnswers: answers,
        failedAttempts: newFailedAttempts,
        feedbackByQuestion: perQuestion,
      },
    ];

    try {
      if (firebaseLogger.isLoggingEnabled()) {
        await firebaseLogger.logEvent(
          "interaction",
          {
            interactionType: "comprehension_check",
            result: allCorrect ? "passed" : "failed",
            failedAttempts: newFailedAttempts,
            answers,
            questions: questions.map((q) => ({
              id: q.id,
              question: q.question,
              options: q.options,
              correctAnswer: q.correctAnswer,
              userAnswer: answers[q.id],
              correctness: perQuestion[q.id],
            })),
            allAnswersCorrect: allCorrect,
            sessionId: gameSessionId,
          },
          "comprehension_check"
        );

        // Force flush for comprehension check as it's critical data
        await firebaseLogger.flushPendingBatches();
      }

      if (allCorrect) {
        onComplete(finalActionLog);
      } else {
        onFailure(finalActionLog);
      }
    } catch (error) {
      console.error("Error submitting comprehension check:", error);
      // Continue flow even if logging fails
      if (allCorrect) {
        onComplete(finalActionLog);
      } else {
        onFailure(finalActionLog);
      }
    }
  }, [
    answers,
    questions,
    onComplete,
    onFailure,
    gameSessionId,
    actionLog,
    failedAttempts,
  ]);

  const handleTryAgain = useCallback(() => {
    // Clear ALL answers
    setAnswers((prev) => {
      const next = { ...prev };
      questions.forEach((q) => {
        next[q.id] = "";
      });
      return next;
    });

    // Clear all feedback
    setFeedbackByQuestion((prev) => {
      const next = { ...prev };
      questions.forEach((q) => {
        next[q.id] = null;
      });
      return next;
    });

    setHasSubmitted(false);

    setActionLog((prevLog) => [
      ...prevLog,
      {
        timestamp: new Date().toISOString(),
        action: "try_again",
      },
    ]);
  }, [questions]);

  const handleReturnToInstructions = useCallback(() => {
    setActionLog((prevLog) => [
      ...prevLog,
      {
        timestamp: new Date().toISOString(),
        action: "return_to_instructions",
      },
    ]);

    if (onReturnToInstructions) {
      onReturnToInstructions();
    }
  }, [onReturnToInstructions]);

  const unansweredCount = questions.filter((q) => !answers[q.id]).length;
  const anyIncorrect =
    hasSubmitted &&
    Object.values(feedbackByQuestion).some((v) => v === "incorrect");
  const allCorrect =
    hasSubmitted &&
    Object.values(feedbackByQuestion).every((v) => v === "correct");

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className="max-w-2xl w-full mx-auto p-6 bg-white rounded-lg shadow-md text-black">
        <h2 className="text-2xl font-bold mb-2 text-black">
          Comprehension Check
        </h2>

        <div className="flex items-center justify-between mb-6">
          <p className="text-sm text-black/80">
            Failed attempts: <span className="font-semibold">{failedAttempts}</span>
          </p>
          <p className="text-xs text-black/60">
            {/* {unansweredCount > 0
              ? `${unansweredCount} unanswered`
              : hasSubmitted
              ? allCorrect
                // ? "All correct"
                // : "Some answers need attention"
              : ""} */}
          </p>
        </div>

        {questions.map((q) => {
          const containerBase =
            "mb-6 p-4 rounded-lg border transition-colors";
          const containerState = "border-gray-200 bg-white";

          return (
            <div key={q.id} className={`${containerBase} ${containerState}`}>
              <p className="font-semibold mb-2 text-black">{q.question}</p>

              <div role="radiogroup" aria-labelledby={`${q.id}-label`}>
                {q.options.map((option) => {
                  const checked = answers[q.id] === option.text;
                  return (
                    <label
                      key={option.key}
                      htmlFor={`${q.id}-${option.key}`}
                      className="flex items-center mb-2 cursor-pointer"
                    >
                      <input
                        type="radio"
                        id={`${q.id}-${option.key}`}
                        name={q.id}
                        value={option.text}
                        checked={checked}
                        onChange={() => handleAnswerChange(q.id, option.text)}
                        className="mr-2"
                      />
                      <span className="text-black">{option.text}</span>
                    </label>
                  );
                })}
              </div>
            </div>
          );
        })}

        <div className="flex flex-col sm:flex-row gap-3">
          <button
            onClick={handleSubmit}
            className="flex-1 bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600 transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
            disabled={unansweredCount > 0 && !hasSubmitted}
            title={
              unansweredCount > 0 && !hasSubmitted
                ? "Please answer all questions"
                : "Submit your answers"
            }
          >
            Submit
          </button>

          {anyIncorrect && (
            <button
              onClick={handleTryAgain}
              className="flex-1 bg-gray-200 text-black py-2 px-4 rounded hover:bg-gray-300 transition-colors"
              title="Clear all answers and try again"
            >
              Try Again
            </button>
          )}

          {onReturnToInstructions && (
            <button
              onClick={handleReturnToInstructions}
              className="flex-1 bg-gray-200 text-black py-2 px-4 rounded hover:bg-gray-300 transition-colors"
              title="Go back to instructions"
            >
              Review Instructions
            </button>
          )}
        </div>

        <div className="mt-4 text-xs text-black/60">
          {/* <p>
            After you submit, questions will be marked{" "}
            <span className="text-green-700 font-medium">Correct</span> or{" "}
            <span className="text-red-700 font-medium">Incorrect</span>. Use
            “Try Again” to clear only the incorrect answers.
          </p> */}
        </div>
      </div>
    </div>
  );
};

export default ComprehensionCheck;
