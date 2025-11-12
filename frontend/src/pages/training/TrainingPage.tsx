import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import "./trainingPage.css";

import type { TrainConfig } from "../../types/lstm";
import FetchService from "../../services/fetchService/FetchService";
import TrainSettings from "../../components/trainSettings/TrainSettings";
import { LSTMChart } from "../../components/lstmChart/LSTMChart";
import SettingsButton from "../../components/settingsButton/SettingsButton";
import LoadingSpinner from "../../components/loadingSpinner/LoadingSpinner";
import ErrorMessage from "../../components/errorMessage/ErrorMessage";
import InfoWidget from "../../components/infoWidget/infoWidget";

const TrainingPage = () => {
   const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(true);

   const handleSettingsButton = () => {
      setIsSettingsOpen(!isSettingsOpen);
   };

   const mutation = useMutation({
      mutationFn: (config: TrainConfig) => FetchService.train(config),
   });

   const onSubmit = (data: TrainConfig) => {
      mutation.mutate(data);
   };

   return (
      <>
         <section className="page training-page">
            <TrainSettings
               settingsButton={{
                  isOpen: isSettingsOpen,
                  handleButton: handleSettingsButton,
               }}
               onSubmit={onSubmit}
            />

            {!mutation.data && !mutation.isPending && (
               <InfoWidget
                  text="Дані тренування відсутні. Проведіть тренування нейромережі"
                  type="data-absence"
               ></InfoWidget>
            )}

            {mutation.isPending && <LoadingSpinner message="Тренування..." />}
            {mutation.isError && (
               <ErrorMessage message={mutation.error.message} />
            )}
            {mutation.isSuccess && <LSTMChart data={mutation.data} />}

            <SettingsButton
               isOpen={isSettingsOpen}
               handleBtn={handleSettingsButton}
               variant="hide-when-open"
            />
         </section>
      </>
   );
};

export default TrainingPage;
