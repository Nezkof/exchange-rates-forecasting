import { useState } from "react";
import TrainSettings from "../../components/trainSettings/TrainSettings";
import "./trainingPage.css";

import ChevronsLeft from "/icons/chevronsLeft.svg";

const TrainingPage = () => {
   const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(true);

   const handleSettingsButton = () => {
      setIsSettingsOpen(!isSettingsOpen);
   };

   return (
      <>
         <section className="page training-page">
            <TrainSettings
               settingsButton={{
                  isSettingsOpen,
                  handleButton: handleSettingsButton,
               }}
            />
            <button
               className={`train-settings__button train-settings__button--abs ${
                  isSettingsOpen ? "train-settings__button--invisible" : ""
               }`}
               onClick={handleSettingsButton}
            >
               <img src={ChevronsLeft} alt="close" />
            </button>
         </section>
      </>
   );
};

export default TrainingPage;
