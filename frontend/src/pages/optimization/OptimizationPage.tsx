import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import "./optimizationPage.css";

import ErrorMessage from "../../components/errorMessage/ErrorMessage";
import LoadingSpinner from "../../components/loadingSpinner/LoadingSpinner";
import SettingsButton from "../../components/settingsButton/SettingsButton";
import FetchService from "../../services/fetchService/FetchService";
import type { OptimizationConfig } from "../../types/portfolio";
import OptimizationSettings from "../../components/optimizationSettings/OptimizationSettings";
import PortfoliosChart from "../../components/portfoliosChart/PortfoliosChart";
import PortfoliosInfo from "../../components/portfoliosInfo/PortfoliosInfo";

const OptimizationPage = () => {
   const [isConfigOpen, setIsConfigOpen] = useState<boolean>(true);

   const handleSettingsButton = () => {
      setIsConfigOpen(!isConfigOpen);
   };

   const mutation = useMutation({
      mutationFn: (config: OptimizationConfig) => FetchService.optimize(config),
   });

   const onSubmit = (data: OptimizationConfig) => {
      mutation.mutate(data);
   };

   return (
      <>
         <section className="page forecasting-page">
            <OptimizationSettings
               settingsButton={{
                  isOpen: isConfigOpen,
                  handleButton: handleSettingsButton,
               }}
               onSubmit={onSubmit}
            />

            {mutation.isPending && <LoadingSpinner message="Portfolio optimization..." />}
            {mutation.isError && <ErrorMessage message={mutation.error.message} />}
            {mutation.isSuccess && <PortfoliosInfo data={mutation.data.portfolios} />}
            {mutation.isSuccess && <PortfoliosChart data={mutation.data} />}

            <SettingsButton
               isOpen={isConfigOpen}
               handleBtn={handleSettingsButton}
               variant="hide-when-open"
            />
         </section>
      </>
   );
};

export default OptimizationPage;
