import "./infoWidget.css";

interface Props {
   text: string | string[];
   type?: "data-absence";
}

const InfoWidget = ({ text, type }: Props) => {
   return (
      <div className="info-widget">
         {type === "data-absence" && (
            <svg
               className="info-widget__icon"
               xmlns="http://www.w3.org/2000/svg"
               viewBox="0 0 24 24"
               stroke="currentColor"
               strokeWidth="1.5"
               strokeLinecap="round"
               strokeLinejoin="round"
            >
               <ellipse cx="12" cy="5" rx="9" ry="3" />
               <path d="M3 5V19A9 3 0 0 0 15 21.84" />
               <path d="M21 5V8" />
               <path d="M21 12L18 17H22L19 22" />
               <path d="M3 12A9 3 0 0 0 14.59 14.87" />
            </svg>
         )}
         {typeof text === "string" ? (
            <p className="info-widget__text">{text}</p>
         ) : (
            <div className="info-widget__text-wrapper">
               {text.map((row, index) => (
                  <p key={index} className="info-widget__text">
                     {row}
                  </p>
               ))}
            </div>
         )}
      </div>
   );
};

export default InfoWidget;
