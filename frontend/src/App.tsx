import "./styles/normalize.css";
import "./styles/fonts.css";
import "./styles/variables.css";
import "./styles/helpers.css";
import "./styles/global.css";

import { BrowserRouter, Route, Routes } from "react-router-dom";
import TrainingPage from "./pages/training/TrainingPage";
import NavigationMenu from "./components/navigationMenu/NavigationMenu";

import { QueryClientProvider } from "@tanstack/react-query";

import "./config/chartConfig";
import queryClient from "./config/queryClientConfig";
import { routes } from "./types/routes";
import ForecastingPage from "./pages/forecasting/ForecastingPage";
import OptimizationPage from "./pages/optimization/OptimizationPage";

const App = () => {
   return (
      <>
         <QueryClientProvider client={queryClient}>
            <BrowserRouter>
               <main>
                  <NavigationMenu />
                  <Routes>
                     <Route path={routes[1].to} element={<TrainingPage />} />
                     <Route path={routes[2].to} element={<ForecastingPage />} />
                     <Route path={routes[3].to} element={<OptimizationPage />} />
                  </Routes>
               </main>
            </BrowserRouter>
         </QueryClientProvider>
      </>
   );
};

export default App;
