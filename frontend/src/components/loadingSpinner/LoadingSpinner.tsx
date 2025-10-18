import "./loadingSpinner.css";

interface LoadingSpinnerProps {
   size?: "small" | "medium" | "large";
   message?: string;
}

const LoadingSpinner = ({ size = "medium", message }: LoadingSpinnerProps) => {
   return (
      <div className="loading-spinner-container">
         <div className={`loading-spinner loading-spinner--${size}`}>
            <div className="spinner"></div>
         </div>
         {message && <p className="loading-spinner-message">{message}</p>}
      </div>
   );
};

export default LoadingSpinner;
