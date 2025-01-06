import './App.css'
import { Routes, Route, BrowserRouter } from "react-router-dom";
import ONNXWithCSV from './ui/ui'
import Two from './Two/Two';
import Three from './Three/Three';


function App() {

  return (
    <>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ONNXWithCSV/>} />
        <Route path="/two" element={<Two/>}/>        
        <Route path="/three" element={<Three/>}/>
      </Routes>
    </BrowserRouter>
    </>
  )
}

export default App
