import ChevronsLeft from "/icons/chevronsLeft.svg";
import "./settingsButton.css";

interface Props {
   isOpen: boolean;
   handleBtn: () => void;
}

const SettingsButton = ({ isOpen, handleBtn }: Props) => {
   return (
      <button
         className={`settings__button ${isOpen && "settings__button--open"}
            `}
         onClick={handleBtn}
      >
         <img src={ChevronsLeft} alt="close" />
      </button>
   );
};

export default SettingsButton;
