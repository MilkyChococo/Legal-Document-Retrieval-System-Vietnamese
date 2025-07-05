import React from "react";
import logo from "../assets/images/logo.png";

const Header = () => {
  return (
    <div className="flex justify-between items-center w-full h-full bg-[#f2f2f2] rounded-b-4xl px-30">
      <div className="flex items-center space-x-4">
        <img src={logo} alt="logo" className="w-[45px] h-[45px]" />
        <div>
          <h2 className="text-[#ba0000] font-bold text-xl">Pháp luật</h2>
          <h1 className="text-[#ffaa1d] font-bold text-xl">VIETNAM</h1>
        </div>
      </div>
      <div className="mr-16">
        <h1 className="text-[#ba0000] font-bold text-xl">
          Chat<span className="text-[#ffaa1d] font-bold text-xl">Law</span>
        </h1>
      </div>
    </div>
  );
};

export default Header;
