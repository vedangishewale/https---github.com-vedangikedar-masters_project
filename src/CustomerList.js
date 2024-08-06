// src/CustomerList.js
import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';

const CustomerList = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [customers, setCustomers] = useState([]);
  const [filteredCustomers, setFilteredCustomers] = useState([]);

  useEffect(() => {
    fetch('/Part2_cluster_1.csv') // Ensure this path is correct
      .then(response => response.text())
      .then(csvText => {
        Papa.parse(csvText, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            setCustomers(results.data);
            setFilteredCustomers(results.data);
          }
        });
      });
  }, []);

  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
  };

  const handleSearchClick = () => {
    const filtered = customers.filter(customer =>
      customer.ID && customer.ID.includes(searchTerm)
    );
    setFilteredCustomers(filtered);
  };

  return (
    <div className="p-4 bg-light rounded shadow-sm mt-4">
      <h3 className="mb-4">Customer List</h3>
      <div className="input-group mb-3">
        <input
          type="text"
          className="form-control"
          placeholder="Search by ID"
          value={searchTerm}
          onChange={handleSearchChange}
        />
        <div className="input-group-append">
          <button className="btn btn-outline-secondary" type="button" onClick={handleSearchClick}>
            Search
          </button>
        </div>
      </div>
      {filteredCustomers.length > 0 ? (
        <ul className="list-group">
          {filteredCustomers.map(customer => (
            <li key={customer.ID || Math.random()} className="list-group-item">
              <strong>ID:</strong> {customer.ID || 'N/A'}
              <br />
              <strong>Beacon Score:</strong> {customer.Beacon_Score || 'N/A'}
              <br />
              <strong>Mortgage Balance:</strong> {customer.Mortgage_Balance || 'N/A'}
              <br />
              <strong>Avg Monthly Transactions:</strong> {customer.Avg_Monthly_Transactions || 'N/A'}
              <br />
              <strong>Has Payroll:</strong> {customer.Has_Payroll || 'N/A'}
              <br />
              <strong>Has Investment:</strong> {customer.Has_Investment || 'N/A'}
              <br />
              <strong>Has Visa:</strong> {customer.Has_Visa || 'N/A'}
              <br />
              <strong>VISA Balance:</strong> {customer.VISA_balance || 'N/A'}
              <br />
              <strong>Has Deposit:</strong> {customer.Has_Deposit || 'N/A'}
              <br />
              <strong>Not Mortgage Balance:</strong> {customer.Not_Mortgage_Balance || 'N/A'}
              <br />
              <strong>Services:</strong> {customer.Services || 'N/A'}
              <br />
              <strong>Tenure In Months:</strong> {customer.Tenure_In_Months || 'N/A'}
              <br />
              <strong>Term In Months:</strong> {customer.TermInMonths || 'N/A'}
              <br />
              <strong>Term To Maturity:</strong> {customer.TermToMaturity || 'N/A'}
              <br />
              <strong>Interest Rate:</strong> {customer.InterestRate || 'N/A'}
              <br />
              <strong>Closing Status:</strong> {customer.Closing_status || 'N/A'}
              <br />
              <strong>Cluster:</strong> {customer.Cluster || 'N/A'}
              <br />
              <strong>Cluster Label:</strong> {customer.Cluster_Label || 'N/A'}
            </li>
          ))}
        </ul>
      ) : (
        <p>No customers found.</p>
      )}
    </div>
  );
};

export default CustomerList;
