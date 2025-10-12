import { NavLink } from "react-router-dom";

import "./navigationMenu.css";
import { routes } from "../../types/routes";

const NavigationMenu = () => {
   return (
      <aside className="side-menu">
         <nav>
            <ul className="navigation-menu">
               {routes.map((link) => (
                  <li key={link.to} className="navigation-menu__item">
                     <NavLink
                        to={link.to}
                        className={({ isActive }) =>
                           isActive
                              ? "navigation-menu__link navigation-menu__link--active"
                              : "navigation-menu__link"
                        }
                     >
                        {link.label}
                     </NavLink>
                  </li>
               ))}
            </ul>
         </nav>
      </aside>
   );
};

export default NavigationMenu;
