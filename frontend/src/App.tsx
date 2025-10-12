import "./styles/normalize.css";
import "./styles/fonts.css";
import "./styles/variables.css";
import "./styles/helpers.css";
import "./styles/global.css";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import TrainingPage from "./pages/training/TrainingPage";
import NavigationMenu from "./components/navigationMenu/NavigationMenu";

const App = () => {
   return (
      <>
         <BrowserRouter>
            <main>
               <NavigationMenu />
               <Routes>
                  <Route path="/" element={<TrainingPage />} />
               </Routes>
            </main>
         </BrowserRouter>
      </>
   );
};

export default App;
