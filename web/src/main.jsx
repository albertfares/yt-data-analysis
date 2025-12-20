import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Intro from "./pages/Intro.jsx";
import User_community from "./pages/User_community.jsx";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Intro />} />
        <Route path="/User_community" element={<User_community />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);