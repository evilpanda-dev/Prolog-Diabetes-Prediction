import React, { useState } from 'react';
import axios from 'axios';
import { Container, Form, Button, Row, Col } from 'react-bootstrap';
import './App.css';

const App = () => {
  const [form, setForm] = useState({
    gender: '',
    age: '',
    hypertension: '',
    heart_disease: '',
    smoking_status: '',
    bmi: '',
    hba1c: '',
    glucose: '',
  });

  const [probability, setProbability] = useState(null);

  const handleChange = e => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async e => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/diabetes-prediction', form);
      setProbability(response.data.probability);
    } catch (error) {
      console.error('Error in sending data: ', error);
    }
  };

  return (
    <Container>
      <Row className="justify-content-md-center">
        <Col md="6">
          <h1 className="text-center">Diabetes Prediction</h1>
          <Form onSubmit={handleSubmit}>
            <Form.Group controlId="gender">
              <Form.Label>Gender</Form.Label>
              <Form.Control type="text" name="gender" onChange={handleChange} />
            </Form.Group>

            <Form.Group controlId="age">
              <Form.Label>Age</Form.Label>
              <Form.Control type="number" name="age" onChange={handleChange} />
            </Form.Group>

            <Form.Group controlId="hypertension">
              <Form.Label>Hypertension</Form.Label>
              <Form.Control type="number" name="hypertension" onChange={handleChange} />
            </Form.Group>

            <Form.Group controlId="heart_disease">
              <Form.Label>Heart Disease</Form.Label>
              <Form.Control type="number" name="heart_disease" onChange={handleChange} />
            </Form.Group>

            <Form.Group controlId="smoking_status">
              <Form.Label>Smoking Status</Form.Label>
              <Form.Control as="select" name="smoking_status" onChange={handleChange}>
                <option value="">Select...</option>
                <option value="never">Never</option>
                <option value="no_info">No Info</option>
                <option value="former">Former</option>
                <option value="current">Current</option>
                <option value="not_current">Not Current</option>
              </Form.Control>
            </Form.Group>

            <Form.Group controlId="bmi">
              <Form.Label>BMI</Form.Label>
              <Form.Control type="number" step="0.01" name="bmi" onChange={handleChange} />
            </Form.Group>

            <Form.Group controlId="hba1c">
              <Form.Label>HbA1c</Form.Label>
              <Form.Control type="number" step="0.01" name="hba1c" onChange={handleChange} />
            </Form.Group>

            <Form.Group controlId="glucose">
              <Form.Label>Glucose</Form.Label>
              <Form.Control type="number" name="glucose" onChange={handleChange} />
            </Form.Group>

            <Button variant="primary" type="submit">
              Submit
            </Button>
          </Form>

          {probability && (
            <p className="mt-4 text-center">
              {probability}
            </p>
          )}
        </Col>
      </Row>
    </Container>
  );
};

export default App;
