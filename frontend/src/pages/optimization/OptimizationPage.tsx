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
import InfoWidget from "../../components/infoWidget/infoWidget";

const OptimizationPage = () => {
   const [isConfigOpen, setIsConfigOpen] = useState<boolean>(false);

   const handleSettingsButton = () => {
      setIsConfigOpen(!isConfigOpen);
   };

   const closeConfig = () => {
      setIsConfigOpen(false);
   };

   const mutation = useMutation({
      mutationFn: (config: OptimizationConfig) => FetchService.optimize(config),
   });

   const onSubmit = (data: OptimizationConfig) => {
      mutation.mutate(data);
      if (window.innerWidth < 768) {
         closeConfig();
      }
   };

   return (
      <>
         <section className="page optimization-page">
            <OptimizationSettings
               settingsButton={{
                  isOpen: isConfigOpen,
                  handleButton: handleSettingsButton,
               }}
               onSubmit={onSubmit}
            />

            {isConfigOpen && (
               <div
                  className="menu-backdrop"
                  onClick={closeConfig}
               ></div>
            )}

            {!mutation.data && !mutation.isPending && (
               <InfoWidget
                  text="Дані оптимізації відсутні. Проведіть оптимізацію інвестиційного портфелю"
                  type="data-absence"
               ></InfoWidget>
            )}

            {mutation.isPending && (
               <LoadingSpinner message="Оптимізація портфеля..." />
            )}
            {mutation.isError && (
               <ErrorMessage message={mutation.error.message} />
            )}
            {mutation.isSuccess && (
               <PortfoliosInfo data={mutation.data.portfolios} />
            )}
            {mutation.isSuccess && <PortfoliosChart data={mutation.data} />}

            <SettingsButton
               isOpen={isConfigOpen}
               handleBtn={handleSettingsButton}
            />
         </section>
      </>
   );
};

export default OptimizationPage;
