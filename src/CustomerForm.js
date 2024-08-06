import React, { useState } from 'react';

const CustomerForm = ({ onAddCustomer }) => {
  const [customer, setCustomer] = useState({
    Beacon_Score: '',
    Mortgage_Balance: '',
    Services: '',
    Avg_Monthly_Transactions: '',
    Has_Payroll: '',
    Has_Investment: '',
    Has_Visa: '',
    VISA_balance: '',
    Has_Deposit: '',
    not_mortgage_lending: '',
    deposit:'',
    Tenure_In_Months: '',
    TermInMonths: '',
    TermToMaturity: '',
    InterestRate: ''
  });

  const [prediction, setPrediction] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setCustomer({ ...customer, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // const response = await axios.post('http://localhost:5000/prediccustomersegmentation', customer);
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
