"use client"
import React, { useEffect, useState } from 'react';
import { firebaseLogger } from '@/services/FirebaseLogger';

interface UserDataModalProps {
  onComplete: () => void;
}

export const UserDataModal: React.FC<UserDataModalProps> = ({ onComplete }) => {
  const [formData, setFormData] = useState({
    prolificId: '',
    age: '',
    gender: '',
    feedback: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [loggingReady, setLoggingReady] = useState(false);

  // Guard: wait until Firebase logging is initialized before allowing submit
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    const checkReady = () => setLoggingReady(firebaseLogger.isLoggingEnabled());
    checkReady();
    if (!loggingReady) {
      interval = setInterval(checkReady, 500);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [loggingReady]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validate required fields
    if (!formData.prolificId.trim() || !formData.age.trim() || !formData.gender) {
      alert('Please fill in all required fields (Prolific ID, Age, and Gender).');
      return;
    }

    setIsSubmitting(true);

    try {
      const userInfo = {
        prolificId: formData.prolificId.trim(),
        age: parseInt(formData.age),
        gender: formData.gender,
        feedback: formData.feedback.trim() || undefined
      };

      await firebaseLogger.addUserInfo(userInfo);
      
      // Force flush for demographic data as it's critical
      await firebaseLogger.flushPendingBatches();
      
      console.log('User data submitted successfully');
      onComplete();
    } catch (error) {
      console.error('Failed to submit user data:', error);
      alert('Failed to submit data. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-8 max-w-md w-full mx-4">
        <h2 className="text-2xl font-bold mb-6 text-center text-black">
          Participant Information
        </h2>
        
        <p className="text-black mb-6 text-center">
          Please provide the following information to participate in the study:
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="prolificId" className="block text-sm font-medium text-black mb-1">
              Prolific ID: <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              id="prolificId"
              value={formData.prolificId}
              onChange={(e) => handleInputChange('prolificId', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-black"
              placeholder="Enter your Prolific ID"
              required
            />
          </div>

          <div>
            <label htmlFor="age" className="block text-sm font-medium text-black mb-1">
              Age: <span className="text-red-500">*</span>
            </label>
            <input
              type="number"
              id="age"
              min="13"
              max="100"
              value={formData.age}
              onChange={(e) => handleInputChange('age', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-black"
              placeholder="Enter your age"
              required
            />
          </div>

          <div>
            <label htmlFor="gender" className="block text-sm font-medium text-black mb-1">
              Gender: <span className="text-red-500">*</span>
            </label>
            <select
              id="gender"
              value={formData.gender}
              onChange={(e) => handleInputChange('gender', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-black"
              required
            >
              <option value="">Please select gender</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="non-binary">Non-binary</option>
              <option value="prefer-not-to-say">Prefer not to say</option>
              <option value="other">Other</option>
            </select>
          </div>

          <div>
            <label htmlFor="feedback" className="block text-sm font-medium text-black mb-1">
              Any feedback about the game? (optional)
            </label>
            <textarea
              id="feedback"
              rows={3}
              value={formData.feedback}
              onChange={(e) => handleInputChange('feedback', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-black"
              placeholder="Share your thoughts..."
            />
          </div>

          {!loggingReady && (
            <div className="text-yellow-700 bg-yellow-50 border border-yellow-200 rounded p-2 text-sm text-center mb-2">
              Initializing connection… please wait a moment.
            </div>
          )}
          <div className="pt-4">
            <button
              type="submit"
              className="w-full px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors disabled:opacity-50"
              disabled={isSubmitting || !loggingReady}
            >
              {isSubmitting ? 'Submitting...' : 'Continue to Study'}
            </button>
          </div>
        </form>

        <p className="text-xs text-black mt-4 text-center">
          Your data helps us understand how players interact with the game.
        </p>
      </div>
    </div>
  );
};

export default UserDataModal;