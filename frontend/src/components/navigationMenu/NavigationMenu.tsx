import { useState } from "react";
import { NavLink } from "react-router-dom";
import "./navigationMenu.css";
import { routes } from "../../types/routes";

const NavigationMenu = () => {
   const [isOpen, setIsOpen] = useState(false);

   const toggleMenu = () => {
      setIsOpen(!isOpen);
   };

   const closeMenu = () => {
      setIsOpen(false);
   };

   return (
      <>
         <button
            className={`menu-toggle ${isOpen ? "menu-toggle--open" : ""}`}
            onClick={toggleMenu}
         >
            <div></div>
            <div></div>
            <div></div>
         </button>

         <aside className={`side-menu ${isOpen ? "side-menu--active" : ""}`}>
            <nav>
               <ul className="navigation-menu">
                  {routes.map((link) => (
                     <li key={link.to} className="navigation-menu__item">
                        <NavLink
                           to={link.to}
                           onClick={closeMenu}
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

         {isOpen && <div className="menu-backdrop" onClick={closeMenu}></div>}
      </>
   );
};

export default NavigationMenu;
