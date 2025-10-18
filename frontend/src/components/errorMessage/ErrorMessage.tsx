import "./errorMessage.css";

interface Props {
   message: string;
}

const ErrorMessage = ({ message }: Props) => {
   return (
      <>
         <div className="error-message">
            <span className="error-message__text">{message}</span>
         </div>
      </>
   );
};

export default ErrorMessage;
