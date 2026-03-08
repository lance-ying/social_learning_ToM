"use client"
import React, { useState } from 'react';
import { firebaseLogger } from '@/services/FirebaseLogger';

interface ConsentFormProps {
  onConsent: () => void;
}

const ConsentForm: React.FC<ConsentFormProps> = ({ onConsent }) => {
  const [hasScrolledToBottom, setHasScrolledToBottom] = useState(false);
  const [doNotRecontact, setDoNotRecontact] = useState(false);
  const [hasConsented, setHasConsented] = useState(false);

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const element = e.currentTarget;
    const isAtBottom = Math.abs(element.scrollHeight - element.scrollTop - element.clientHeight) < 10;
    if (isAtBottom && !hasScrolledToBottom) {
      setHasScrolledToBottom(true);
    }
  };

  const handleDoNotRecontactChange = (checked: boolean) => {
    setDoNotRecontact(checked);
    
    // Log to Firebase
    firebaseLogger.logConsentPreference({
      doNotRecontact: checked,
      timestamp: new Date().toISOString()
    });
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className="max-w-4xl w-full mx-auto bg-white rounded shadow p-8">
        <h1 className="text-2xl font-bold mb-6 text-center text-black">Research Study Consent Form</h1>
        
        <div 
          className="h-[500px] overflow-y-auto border border-gray-300 p-6 mb-6 bg-white"
          onScroll={handleScroll}
        >
          <div className="text-sm text-black leading-relaxed">
            <p className="font-bold mb-3">Key Information</p>
            <p className="mb-4">
              The following is a short summary of this study to help you decide whether to be a part of this study. More detailed information is listed later in this form.
            </p>

            <p className="font-bold mb-2">Why am I being invited to take part in a research study?</p>
            <p className="mb-4">
              We invite you to take part in a research study because you have met the eligibility criteria for this study. Specifically, you are an adult in the appropriate age range. As a volunteer, you have been asked to participate in this study because your results may help us to understand the brain more fully.
            </p>

            <p className="font-bold mb-2">What should I know about a research study?</p>
            <p className="mb-1">• Someone will explain this research study to you.</p>
            <p className="mb-1">• Whether or not you take part is up to you.</p>
            <p className="mb-1">• Your participation is completely voluntary.</p>
            <p className="mb-1">• You can choose not to take part.</p>
            <p className="mb-1">• You can agree to take part and later change your mind.</p>
            <p className="mb-1">• Your decision will not be held against you.</p>
            <p className="mb-1">• Your refusal to participate will not result in any consequences or any loss of benefits that you are otherwise entitled to receive.</p>
            <p className="mb-4">• You can ask all the questions you want before you decide.</p>

            <p className="font-bold mb-2">Why is this research being done?</p>
            <p className="mb-4">
              The purpose of this study is to investigate the way people make decisions and learn about the relationships between stimuli, actions and rewards.
            </p>

            <p className="font-bold mb-2">How long will the research last and what will I need to do?</p>
            <p className="mb-4">
              Stimuli will be presented on the computer and you will be asked to choose between different actions after which you may receive feedback. You may also be asked to fill out short questionnaires. This will take less than 1 hour.
            </p>

            <p className="font-bold mb-2">Is there any way being in this study could be bad for me?</p>
            <p className="mb-4">
              Some of the questionnaires contain questions about sensitive topics such as alcohol/drug use, medical/psychiatric history, etc., and could cause you to feel uncomfortable or upset. If you do not feel comfortable, you can skip questions or stop the questionnaires at any time.
            </p>

            <p className="font-bold mb-2">Will being in this study help me in any way?</p>
            <p className="mb-4">
              This study provides no benefits to you individually. The study provides important information about the nature of learning and decision making. This has potential implications for public domains, such as healthcare policy and the justice system, where understanding the processes governing choice (e.g., of insurance, medical care, illegal substances) can lead to more psychologically effective public policy.
            </p>

            <p className="font-bold mb-2">What happens if I do not want to be in this research?</p>
            <p className="mb-4">
              Participation in research is completely voluntary. You can decide to participate, not participate, or discontinue participation at any time without penalty or loss of benefits to which you are otherwise entitled. Your alternative to participating in this research study is to not participate.
            </p>

            <p className="font-bold mb-3 mt-6">Detailed Information</p>
            <p className="mb-4">
              The following is more detailed information about this study in addition to the information listed above.
            </p>

            <p className="font-bold mb-2">What is the purpose of this research?</p>
            <p className="mb-4">
              The study will use measures of your behavior (your choices and response times) to understand the psychological mechanisms underlying learning and decision making.
            </p>

            <p className="font-bold mb-2">How long will I take part in this research?</p>
            <p className="mb-4">
              The study duration will take about 15 minutes (not including breaks).
            </p>

            <p className="font-bold mb-2">What happens if I say yes, I want to be in this research?</p>
            <p className="mb-4">
              After providing informed consent and receiving instructions, the main part of the study will begin. Stimuli will be presented on your personal computer and you will be asked to choose between different actions after which you may receive feedback. Before or after completing the experimental task, you may be asked to respond to one or more questionnaires that provide us with information about you and your thinking style. The study will take less than 1 hour total. Because the study is taking place through a web interface, you will not be directly interacting with study personnel, though you may contact them at any time (see below). After the study is over, you will receive a debriefing form. You may be contacted for future research.
            </p>
            <p className="mb-4">
              In some cases, we may be interested in re-contacting you for additional information or to participate in a follow-up experiment. If we do, your participation is completely optional and you would be compensated appropriately for your time.
            </p>

            <p className="font-bold mb-2">What happens if I say yes, but I change my mind later?</p>
            <p className="mb-4">
              You can leave the research at any time; it will not be held against you. You will still receive your base payment for the completed portion of the study, though you may forego any bonus payments associated with task performance if the task was not completed. If you choose to withdraw from the study, we will ask you for permission to continue using any data that were already collected. If you do not give permission, we will delete the data.
            </p>

            <p className="font-bold mb-2">Is there any way being in this study could be bad for me? (Detailed Risks)</p>
            <p className="mb-4">
              There are some mild risks you might experience from being in this study, as detailed below. Please note that we take every effort to minimize risks for you.
            </p>
            <p className="mb-4">
              Some of the questionnaires contain questions about sensitive topics such as medical/psychiatric history, etc., and could cause you to feel uncomfortable or upset. If you do not feel comfortable, you can skip questions or stop the questionnaires at any time.
            </p>
            <p className="mb-4">
              We will do our best to protect your data and samples during storage and when they are shared. However, there remains a possibility that someone could identify you. There is also the possibility that people who are not supposed to might access your data. In either case, we cannot reduce the risk to zero.
            </p>

            <p className="font-bold mb-2">If I take part in this research, how will my privacy be protected? What happens to the information you collect?</p>
            <p className="mb-4">
              Efforts will be made to limit the use and disclosure of your Personal Information, including survey responses, to people who have a need to review this information. We cannot promise complete secrecy. Organizations that may inspect and copy your information include the IRB and other representatives of this organization.
            </p>
            <p className="mb-4">
              Your participation in this study will remain confidential, and your identity will not be stored with your data. Your responses will be assigned a code number, and the list connecting your name with this number will be kept in a locked room or in a password protected computer file.
            </p>
            <p className="mb-4">
              If identifiers are removed from your identifiable private information that are collected during this research, that information could be used for future research studies or distributed to another investigator for future research studies without your additional informed consent.
            </p>

            <p className="font-bold mb-2">Can I be removed from the research without my OK?</p>
            <p className="mb-4">
              The person in charge of the research study or the sponsor can remove you from the research study without your approval. Possible reasons for removal include discovering a previously unidentified ineligibility or failure to comply with task instructions.
            </p>
            <p className="mb-4">
              We will tell you about any new information that may affect your health, welfare, or choice to stay in the research.
            </p>

            <p className="font-bold mb-2">Compensation</p>
            <p className="mb-4">
              You will receive cash payment for this study at the rate of $10/hour. For some studies, you may win additional cash prizes or gift cards (typically within the range of $1-$5) based on your performance in the study. These prizes will be paid to you immediately after your participation.
            </p>

            <p className="font-bold mb-2">You may not be told everything or may be misled</p>
            <p className="mb-4">
              As part of this research design, you may not be told or may be misled about the purpose or procedures of this research. However, the purpose or procedures of the research will be disclosed to you following your participation.
            </p>

            <p className="font-bold mb-2">Who can I talk to?</p>
            <p className="mb-4">
              If you have questions, concerns, or complaints, or think the research has hurt you, talk to the research team by contacting Ryan Truong at{' '}
              <a href="mailto:truongtruong@g.harvard.edu" className="text-blue-600 underline">
                truongtruong@g.harvard.edu
              </a>. You may also contact the Principal Investigator, Samuel Gershman, at{' '}
              <a href="mailto:gershman@fas.harvard.edu" className="text-blue-600 underline">
                gershman@fas.harvard.edu
              </a>.
            </p>
            <p className="mb-4">
              This research has been reviewed and approved by the Harvard University Area Institutional Review Board ("IRB"). Learn more about the IRB and your rights as a participant on the IRB's For Research Participants webpage. You may contact the IRB at (617) 496-2847 or{' '}
              <a href="mailto:cuhs@harvard.edu" className="text-blue-600 underline">
                cuhs@harvard.edu
              </a>{' '}
              if:
            </p>
            <p className="mb-1">• Your questions, concerns, or complaints are not being answered by the research team.</p>
            <p className="mb-1">• You cannot reach the research team.</p>
            <p className="mb-1">• You want to talk to someone besides the research team.</p>
            <p className="mb-1">• You have questions about your rights as a research subject.</p>
            <p className="mb-4">• You want to get information or provide input about this research.</p>

            <p className="font-bold mb-2 mt-4">Agreement:</p>
            <p className="mb-4">
              The nature and purpose of this research have been sufficiently explained and I agree to participate in this study. I understand that I am free to withdraw at any time without incurring any penalty.
            </p>
            <p className="mb-4">
              Please consent by clicking the button below to continue. Otherwise, please exit the study at this time.
            </p>
          </div>
        </div>

        <div className="flex flex-col items-center space-y-4">
          {!hasScrolledToBottom && (
            <p className="text-sm text-gray-600">
              Please scroll to the bottom of the consent form to continue
            </p>
          )}
          
          <div className="flex flex-col space-y-3">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="consent"
                checked={hasConsented}
                onChange={(e) => setHasConsented(e.target.checked)}
                disabled={!hasScrolledToBottom}
                className={`w-5 h-5 mr-3 ${hasScrolledToBottom ? 'cursor-pointer' : 'cursor-not-allowed opacity-50'}`}
              />
              <label 
                htmlFor="consent" 
                className={`font-semibold text-black ${hasScrolledToBottom ? 'cursor-pointer' : 'cursor-not-allowed opacity-50'}`}
              >
                I consent to participate in this study
              </label>
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                id="doNotRecontactBottom"
                checked={doNotRecontact}
                onChange={(e) => handleDoNotRecontactChange(e.target.checked)}
                disabled={!hasScrolledToBottom}
                className={`w-5 h-5 mr-3 ${hasScrolledToBottom ? 'cursor-pointer' : 'cursor-not-allowed opacity-50'}`}
              />
              <label 
                htmlFor="doNotRecontactBottom" 
                className={`text-black ${hasScrolledToBottom ? 'cursor-pointer' : 'cursor-not-allowed opacity-50'}`}
              >
                I do not wish to be recontacted
              </label>
            </div>
          </div>

          <button
            onClick={onConsent}
            disabled={!hasScrolledToBottom || !hasConsented}
            className={`px-8 py-3 rounded font-semibold transition-colors ${
              hasScrolledToBottom && hasConsented
                ? 'bg-green-600 hover:bg-green-700 text-white cursor-pointer'
                : 'bg-gray-300 text-gray-500 cursor-not-allowed'
            }`}
          >
            Continue
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConsentForm;
