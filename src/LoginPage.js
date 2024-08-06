import './LoginPage.css';
import React from 'react';
import { useNavigate } from 'react-router-dom';
import './LoginPage.css';  // Create and import a custom CSS file

const LoginPage = () => {
  const navigate = useNavigate();

  const handleNavigate = (path) => {
    navigate(path);
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <h2 className="login-title">Welcome</h2>
        <p className="login-subtitle">Please select an option to proceed</p>
        <button
          className="btn btn-primary btn-block mt-3"
          onClick={() => handleNavigate('/interest-rate-prediction')}
        >
          Interest Rate Prediction
        </button>
        <button
          className="btn btn-secondary btn-block mt-3"
          onClick={() => handleNavigate('/customer-renewal-prediction')}
        >
          Customer Renewal Prediction
        </button>
      </div>
    </div>
  );
};

export default LoginPage;
