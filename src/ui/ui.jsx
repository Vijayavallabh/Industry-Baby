import React, { useState } from 'react';
import * as ort from 'onnxruntime-web';
import Papa from 'papaparse'; // Library to parse CSV files
import './ui.css';
import Modal from '../Modal/Modal';  // Import the modal component
import Navbar from '../Navbar/Navbar';
// import axios from "axios";




const ONNXWithCSV = () => {
    const [fileData, setFileData] = useState(null); // Parsed CSV data
    const [predictions, setPredictions] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isModalOpen, setIsModalOpen] = useState(false);  // State to control modal visibility

    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        if (file) {
            // Parse the CSV file
            Papa.parse(file, {
                complete: (result) => {
                    setFileData(result.data); // Set the parsed data
                },
                header: false, // Set to true if CSV has headers
                skipEmptyLines: true,
            });
        }
    };

    const runModel = async () => {
        if (!fileData || fileData.length === 0) {
            alert("Please upload a valid CSV file with data.");
            return;
        }

        try {
            setIsLoading(true);

            // Load the ONNX model
            const session = await ort.InferenceSession.create('/model.onnx');

            // Run predictions for each row in the CSV
            const results = [];
            for (const row of fileData) {
                const inputArray = row.map(Number); // Convert row data to numbers
                const inputTensor = new ort.Tensor("float32", new Float32Array(inputArray), [1, inputArray.length]);

                // Prepare feeds
                const feeds = { input: inputTensor }; // Replace 'input' with the actual input name in your model

                // Run inference
                const result = await session.run(feeds);

                // Extract output
                const outputName = Object.keys(result)[0];
                const outputData = result[outputName].data;

                results.push(outputData);
            }

            setPredictions(results);
            setIsModalOpen(true);  // Open the modal to show predictions
            setIsLoading(false);
        } catch (error) {
            console.error("Error running ONNX model:", error);
            setIsLoading(false);
        }
    };

    const closeModal = () => {
        setIsModalOpen(false);  // Close the modal
    };

// const FileUploader = () => {
//   const [file, setFile] = useState(null);
//   const [response, setResponse] = useState("");

//   const handleFileChange = (event) => {
//     setFile(event.target.files[0]);
//   };

//   const uploadFile = async () => {
//     if (!file) {
//       alert("Please select a file first!");
//       return;
//     }

//     const formData = new FormData();
//     formData.append("file", file);

//     try {
//       const res = await axios.post("http://127.0.0.1:5000/upload", formData, {
//         headers: {
//           "Content-Type": "multipart/form-data",
//         },
//       });
//       setResponse(res.data.message || "File uploaded successfully!");
//     } catch (err) {
//       console.error(err);
//       setResponse("An error occurred while uploading the file.");
//     }
//   };


    return (
        <>
            
            <div className="main">
                <Navbar/>
                <div className="grid-container">
                    <div className="plane">
                        <div className="grid"></div>
                        <div className="glow"></div>
                    </div>
                    <div className="plane">
                        <div className="grid"></div>
                        <div className="glow"></div>
                    </div>
                </div>
                <h1 className="h1">Risk assessment solution</h1>
                <div className="multipleusers">
                    <h3 className="h3">Upload CSV File</h3>
                    <input
                        className="svginput"
                        type="file"
                        accept=".csv"
                        onChange={handleFileUpload}
                    />
                    <div className="predict">
                    <a href="#" className="a">
                        <span></span>
                        <span></span>
                        <span></span>
                        <span></span>
                        <button className="predictbtn" onClick={runModel} disabled={isLoading || !fileData}>
                            {isLoading ? "Running..." : "Predict"}
                        </button>
                    </a>
                </div>
                </div>

                
                <Modal isOpen={isModalOpen} onClose={closeModal} predictions={predictions} />
            </div>
        </>
    );
};

export default ONNXWithCSV;
