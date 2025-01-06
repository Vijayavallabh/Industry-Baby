import React, { useEffect, useState } from "react";
import "./Navbar.css";
import { useLocation, useNavigate } from "react-router-dom";


const Navbar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [activeDiv, setActiveDiv] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    if (location.pathname === "/") {
      setActiveDiv("home");
    } else if (location.pathname === "/secondpage") {
      setActiveDiv("secondpage");
    } else if (location.pathname === "/thirdpage") {
      setActiveDiv("thirdpage");
    }
  }, [location.pathname]);

  const handleClick = () => {
    setActiveDiv("secondpage");
    navigate("/two");
    setMenuOpen(false);
  };

  const homeclick = () => {
    setActiveDiv("home");
    navigate("/");
    setMenuOpen(false);
  };



  const handlethree = () => {
    setActiveDiv("handlethree");
    navigate("/three");
    setMenuOpen(false);
  };

 
  return (
    <div className="navbar">
            <div
            className= "home"
            onClick={homeclick}
            >
            Home
            </div>
            <div
            className="second"
            onClick={handleClick}
            >
            Second page
            </div>
            <div
            className="third"
            onClick={handlethree}
            >
            Third page
            </div>
    </div>
  );
};

export default Navbar;
