import { useEffect, useState } from "react";
import TrainSettings from "../../components/trainSettings/TrainSettings";
import "./trainingPage.css";

import FetchService from "../../services/fetchService/FetchService";
import type { SettingsConfig } from "../../types/lstm";
import { useMutation } from "@tanstack/react-query";
import { LSTMChart } from "../../components/lstmChart/LSTMChart";
import SettingsButton from "../../components/settingsButton/SettingsButton";
import LoadingSpinner from "../../components/loadingSpinner/LoadingSpinner";
import ErrorMessage from "../../components/errorMessage/ErrorMessage";

const TrainingPage = () => {
   const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(true);

   const handleSettingsButton = () => {
      setIsSettingsOpen(!isSettingsOpen);
   };

   const mutation = useMutation({
      mutationFn: (config: SettingsConfig) => FetchService.trainModel(config),
   });

   const onSubmit = (data: SettingsConfig) => {
      mutation.mutate(data);
   };

   useEffect(() => {
      console.log("Mutation status:", {
         isSuccess: mutation.isSuccess,
         isPending: mutation.isPending,
         isError: mutation.isError,
         data: mutation.data,
         error: mutation.error,
      });
   }, [mutation.isSuccess, mutation.isPending, mutation.isError, mutation.data]);
   return (
      <>
         <section className="page training-page">
            <TrainSettings
               settingsButton={{
                  isSettingsOpen,
                  handleButton: handleSettingsButton,
               }}
               onSubmit={onSubmit}
            />

            {mutation.isPending && <LoadingSpinner message="Training model..." />}
            {mutation.isError && <ErrorMessage message={mutation.error.message} />}
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
