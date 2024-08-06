import React from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import CustomerForm from './CustomerForm';
import CustomerList from './CustomerList';
import LoginPage from './LoginPage';
import InterestRatePredictionPage from './InterestRatePredictionPage';
import './App.css';

const App = () => {
  return (
    <Router>
      <nav className="navbar navbar-expand-lg navbar-light bg-light">
        <div className="container">
          <Link className="navbar-brand" to="/">Home</Link>
        </div>
      </nav>
      <Routes>
        <Route path="/" element={<LoginPage />} />
        <Route path="/interest-rate-prediction" element={<InterestRatePredictionPage />} />
        <Route path="/customer-renewal-prediction" element={
          <div className="container mt-5">
            <div className="row">
              <div className="col-md-6">
                <CustomerForm />
              </div>
              <div className="col-md-6">
                <CustomerList />
              </div>
            </div>
          </div>
        } />
      </Routes>
    </Router>
  );
};

export default App;
