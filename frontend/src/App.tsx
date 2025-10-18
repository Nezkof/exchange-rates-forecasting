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

const App = () => {
   return (
      <>
         <QueryClientProvider client={queryClient}>
            <BrowserRouter>
               <main>
                  <NavigationMenu />
                  <Routes>
                     <Route path="/" element={<TrainingPage />} />
                  </Routes>
               </main>
            </BrowserRouter>
         </QueryClientProvider>
      </>
   );
};

export default App;
