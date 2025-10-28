import ChevronsLeft from "/icons/chevronsLeft.svg";
import "./settingsButton.css";

interface Props {
   isOpen: boolean;
   handleBtn: () => void;
   variant?: "default" | "hide-when-open";
}

const SettingsButton = ({ isOpen, handleBtn, variant = "default" }: Props) => {
   if (variant === "hide-when-open" && isOpen) {
      return null;
   }

   return (
      <button
         className={`settings__button ${
            isOpen ? "settings__button--open" : "settings__button--abs"
         }`}
         onClick={handleBtn}
      >
         <img src={ChevronsLeft} alt="close" />
      </button>
   );
};

export default SettingsButton;
