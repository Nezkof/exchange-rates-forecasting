interface Props {
   isOpen: boolean;
   handleBtn: () => void;
   variant?: "default" | "hide-when-open";
}

import ChevronsLeft from "/icons/chevronsLeft.svg";

const SettingsButton = ({ isOpen, handleBtn, variant = "default" }: Props) => {
   if (variant === "hide-when-open" && isOpen) {
      return null;
   }

   return (
      <button
         className={`train-settings__button ${
            isOpen ? "train-settings__button--open" : "train-settings__button--abs"
         }`}
         onClick={handleBtn}
      >
         <img src={ChevronsLeft} alt="close" />
      </button>
   );
};

export default SettingsButton;
