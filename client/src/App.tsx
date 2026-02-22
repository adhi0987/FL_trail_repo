// client/src/App.tsx
import { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { generatePersonaData } from './dataGenerator';
import './App.css';

const API_URL = "http://localhost:8000";

function App() {
  const [globalRound, setGlobalRound] = useState(0);
  const [trainingStatus, setTrainingStatus] = useState("Idle");
  const [localFPR, setLocalFPR] = useState<number | null>(null);

  const buildModel = () => {
    const model = tf.sequential();
    // Input: X, Y, Z accelerometer data
    model.add(tf.layers.dense({ units: 10, inputShape: [3], activation: 'relu' }));
    // Output: Normal, Hard Sit, Fall
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    return model;
  };

  const runFederatedRound = async (persona: 'athlete' | 'senior' | 'average') => {
    setTrainingStatus(`Fetching Global Model...`);
    
    // 1. Fetch Global Weights
    const res = await fetch(`${API_URL}/api/model/global`);
    const globalData = await res.json();
    setGlobalRound(globalData.round);

    // 2. Load weights into local model
    const model = buildModel();
    if (globalData.weights.length > 0) {
      const weightTensors = globalData.weights.map((w: any) => tf.tensor(w));
      model.setWeights(weightTensors);
    }

    // 3. Generate Local Data & Train (Personalization)
    setTrainingStatus(`Training Local Model (${persona})...`);
    const { xs, ys } = generatePersonaData(persona);
    
    await model.fit(xs, ys, {
      epochs: 5,
      callbacks: {
        onEpochEnd: (epoch, logs) => console.log(`Epoch ${epoch}: loss = ${logs?.loss}`)
      }
    });

    // Calculate a mock FPR (In reality, run model.predict on a test set)
    const simulatedFPR = persona === 'athlete' ? Math.random() * 0.1 : Math.random() * 0.4;
    setLocalFPR(simulatedFPR);

    // 4. Extract updated weights and send to server
    setTrainingStatus("Uploading Weight Deltas...");
    const updatedWeights = model.getWeights().map(w => w.arraySync());
    
    await fetch(`${API_URL}/api/model/update`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        client_id: `client_${Math.floor(Math.random() * 1000)}`,
        weights: updatedWeights,
        local_fpr: simulatedFPR
      })
    });

    setTrainingStatus("Idle - Waiting for next round");
  };

  return (
    <div className="card" style={{ padding: '2rem', maxWidth: '800px', margin: 'auto' }}>
      <h1>Federated Fall Detection</h1>
      
      <div style={{ background: '#333', padding: '1rem', borderRadius: '8px', marginBottom: '2rem' }}>
        <h2>Global Server State</h2>
        <p>Current Communication Round: <strong>{globalRound}</strong></p>
        <p>Status: {trainingStatus}</p>
      </div>

      <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
        <div style={{ border: '1px solid #555', padding: '1rem', borderRadius: '8px' }}>
          <h3>Simulate Client A (Athlete)</h3>
          <p>High impact movements. High False Positive risk.</p>
          <button onClick={() => runFederatedRound('athlete')}>Run Local Training</button>
          {localFPR !== null && <p>Local FPR: {(localFPR * 100).toFixed(2)}%</p>}
        </div>

        <div style={{ border: '1px solid #555', padding: '1rem', borderRadius: '8px' }}>
          <h3>Simulate Client B (Senior)</h3>
          <p>Slow movements. Needs highly sensitive fall detection.</p>
          <button onClick={() => runFederatedRound('senior')}>Run Local Training</button>
        </div>
      </div>
    </div>
  );
}

export default App;