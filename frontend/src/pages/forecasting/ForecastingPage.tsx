import { useMutation } from "@tanstack/react-query";
import "./forecastingPage.css";
import FetchService from "../../services/fetchService/FetchService";
import { useState } from "react";
import type { ForecastConfig } from "../../types/lstm";
import SettingsButton from "../../components/settingsButton/SettingsButton";
import LoadingSpinner from "../../components/loadingSpinner/LoadingSpinner";
import ErrorMessage from "../../components/errorMessage/ErrorMessage";
import { LSTMChart } from "../../components/lstmChart/LSTMChart";
import ForecastSettings from "../../components/forecastSettings/ForecastSettings";
import MetricsTable from "../../components/metricsTable/MetricsTable";

const ForecastingPage = () => {
   const [isConfigOpen, setIsConfigOpen] = useState<boolean>(true);

   const handleSettingsButton = () => {
      setIsConfigOpen(!isConfigOpen);
   };

   const mutation = useMutation({
      mutationFn: (config: ForecastConfig) => FetchService.forecast(config),
   });

   const onSubmit = (data: ForecastConfig) => {
      mutation.mutate(data);
   };

   return (
      <>
         <section className="page forecasting-page">
            <ForecastSettings
               settingsButton={{
                  isOpen: isConfigOpen,
                  handleButton: handleSettingsButton,
               }}
               onSubmit={onSubmit}
            />

            {mutation.isPending && <LoadingSpinner message="Прогнозування..." />}
            {mutation.isError && <ErrorMessage message={mutation.error.message} />}
            {mutation.isSuccess && <LSTMChart data={mutation.data} />}
            {mutation.isSuccess && <MetricsTable data={mutation.data.metrics} />}

            <SettingsButton
               isOpen={isConfigOpen}
               handleBtn={handleSettingsButton}
               variant="hide-when-open"
            />
         </section>
      </>
   );
};

export default ForecastingPage;
