import React from 'react';
import './Modal.css'; 

const Modal = ({ isOpen, onClose, predictions }) => {
    if (!isOpen) return null;

    return (
        <div className="modal-overlay">
            <div className="modal-content">
                <h3>Predictions</h3>
                {predictions.map((prediction, index) => (
                    <p key={index}>Row {index + 1}: {JSON.stringify(prediction)}</p>
                ))}
                <button onClick={onClose} className="modal-close-btn">Close</button>
            </div>
        </div>
    );
};

export default Modal;
