"use client"
import React from 'react';

interface DebriefProps {
  totalPoints: number;
}

const Debrief: React.FC<DebriefProps> = ({ totalPoints }) => {
  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className="max-w-4xl w-full mx-auto bg-white rounded shadow p-8">
        <h1 className="text-2xl font-bold mb-6 text-center text-black">Thank You!</h1>

        <div className="mb-6 text-center">
          <p className="text-lg mb-2 text-black">Your participation is greatly appreciated.</p>
          <p className="text-xl font-semibold mb-2 text-black">Completion code: <span className="text-green-600">C1KX63GD</span></p>
          <p className="text-lg text-black">Total points earned: <span className="font-semibold">{totalPoints}</span></p>
        </div>

        <hr className="my-6 border-gray-300" />

        <div className="text-sm text-black leading-relaxed space-y-4">
          <h2 className="text-xl font-bold mb-4">Study Debriefing</h2>

          <div>
            <p className="font-bold mb-2">What was this study about?</p>
            <p>
              Decision making is a ubiquitous aspect of life. Using tasks like the one you just completed, we are examining the factors that go into making a decision and learning from reward feedback.
            </p>
          </div>

          <div>
            <p className="font-bold mb-2">How was the study conducted?</p>
            <p>
              We have asked you to make choices between actions that vary in their future reward. By isolating different variables that influence these decisions, and how you learn from feedback, we can better understand how people perform complex decision-making. For example, sometimes action sequences are "chunked" into a single action that can be selected more quickly than selecting each action sequentially. We would like to understand how humans decide which action sequences to chunk.
            </p>
          </div>

          <div>
            <p className="font-bold mb-2">What was the hypothesis?</p>
            <p>
              A fundamental goal of our research is to understand the cognitive factors that influence decision-making. We are studying these factors by presenting people with choices that vary in specific ways and seeing which factors make a difference.
            </p>
          </div>

          <div>
            <p className="font-bold mb-2">Why is this study important?</p>
            <p>
              By comparing answers on these important factors, we learn about what factors affect decision making. This has potential implications for public domains, such as healthcare policy and the justice system, where understanding the processes governing choice (e.g., of insurance, medical care, illegal substances) can lead to more psychologically effective public policy.
            </p>
          </div>

          <div>
            <p className="font-bold mb-2">References:</p>
            <p className="text-sm italic">
              Gershman, S.J. (2015). Reinforcement learning and causal models. In M. Waldmann, Ed, Oxford Handbook of Causal Reasoning. Oxford University Press.
            </p>
          </div>

          <div className="mt-6 pt-4 border-t border-gray-300">
            <p className="font-bold mb-2">How to contact the researcher:</p>
            <p className="mb-4">
              If you have questions or concerns about your participation or payment, or want to request a summary of research findings, please contact the researcher: Ryan Truong,{' '}
              <a href="mailto:truongtruong@g.harvard.edu" className="text-blue-600 underline">
                truongtruong@g.harvard.edu
              </a>. You may also contact the Principal Investigator, Samuel Gershman, at{' '}
              <a href="mailto:gershman@fas.harvard.edu" className="text-blue-600 underline">
                gershman@fas.harvard.edu
              </a>.
            </p>

            <p className="font-bold mb-2">Whom to contact about your rights as a participant in this research:</p>
            <p>
              For questions, concerns, suggestions, or complaints that have not been or cannot be addressed by the researcher, or to report research-related harm, please contact the Committee on the Use of Human Subjects, Richard A. and Susan F. Smith Campus Center, 1350 Massachusetts Avenue, Suite 935, Cambridge, MA 02138; Email:{' '}
              <a href="mailto:cuhs@harvard.edu" className="text-blue-600 underline">
                cuhs@harvard.edu
              </a>. Phone: 617-496-2847.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Debrief;
