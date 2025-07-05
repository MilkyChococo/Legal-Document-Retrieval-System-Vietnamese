import React from "react";
import Header from "./components/Header";
import Chat from "./components/Chat";

const App = () => {
  return (
    <div className="w-full h-screen bg-[#595959]">
      <div className="h-[13%]">
        <Header />
      </div>
      <div className="h-[87%] flex justify-center items-center">
        <Chat />
      </div>
    </div>
  );
};

export default App;
