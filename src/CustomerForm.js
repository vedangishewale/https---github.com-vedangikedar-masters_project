import React, { useState } from 'react';

const CustomerForm = ({ onAddCustomer }) => {
  const [customer, setCustomer] = useState({
    Beacon_Score: 0,
    Mortgage_Balance: 0.0,
    Services: 0,
    Avg_Monthly_Transactions: 0,
    Has_Payroll: 0,
    Has_Investment: 0,
    Has_Visa: 0,
    VISA_balance: 0.0,
    Has_Deposit: 0,
    not_mortgage_lending: 0.0,
    deposit: 0.0,
    Tenure_In_Months: 0,
    TermInMonths: 0,
    TermToMaturity: 0,
    InterestRate: 0.0
  });

  const [prediction, setPrediction] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;    
    // Cast to appropriate type based on the field
    const newValue = (name === 'Mortgage_Balance' || name === 'VISA_Balance' ||
                      name === 'Non_Mortgage_Lending' || name === 'Deposit' ||
                      name === 'InterestRate') ? parseFloat(value) : parseInt(value, 10);
    setCustomer({ ...customer, [name]: newValue });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://127.0.0.1:5000/prediccustomersegmentation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(customer)
      });
      const data = await response.json();
      console.log(data)
      setPrediction(data);
      onAddCustomer({ ...customer, renewalStatus: data.renewal, renewallabel: data.label });
    } catch (error) {
      console.error('Error making prediction:', error);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit} className="p-4 bg-light rounded shadow-sm">
        <h3 className="mb-4">Customer Prediction</h3>
        <div className="row">
          {Object.keys(customer).map((key) => (
            <div key={key} className="form-group col-md-6">
              <label>{key.replace(/([A-Z])/g, ' $1').trim().replace(/^./, key[0].toUpperCase()).replace('Visa ', 'VISA ').replace('Not Mortgage ', 'Not Mortgage ').replace('Avg ', 'Avg ').replace('Tenure ', 'Tenure ').replace('Term ', 'Term ').replace('Interest ', 'Interest ') }:</label>
              <input
                type="text"
                name={key}
                value={customer[key]}
                onChange={handleChange}
                className="form-control"
              />
            </div>
          ))}
        </div>
        <button type="submit" className="btn btn-primary mt-3">Predict Renewal Status</button>
      </form>
      {prediction && (
        <div className="alert alert-info mt-4">
          The customer is predicted to be: <strong>{prediction.renewal}</strong>
          <br />
          Cluster: <strong>{prediction.label}</strong>
        </div>
      )}
    </div>
  );
};

export default CustomerForm;
