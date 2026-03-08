interface DebugPasswordModalProps {
  onSubmit: (password: string) => void;
  onClose: () => void;
}

const DebugPasswordModal: React.FC<DebugPasswordModalProps> = ({ onSubmit, onClose }) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white p-6 rounded-lg">
        <h2 className="text-xl mb-4 text-black font-bold">Enter Debug Password</h2>
        <form onSubmit={(e) => {
          e.preventDefault();
          const password = (e.target as any).password.value;
          onSubmit(password);
        }}>
          <input
            type="password"
            name="password"
            className="border p-2 mb-4 w-full text-black"
            autoFocus
          />
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Submit
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default DebugPasswordModal; 