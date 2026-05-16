import React, { useState } from 'react';
import PredictForm from './components/PredictForm';
import ResultCard from './components/ResultCard';
import CareerPath from './components/CareerPath';
import { fetchPrediction } from './api/predict';
import { AlertTriangle } from 'lucide-react';

function App() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async (formData) => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await fetchPrediction(formData);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div style={{ textAlign: 'center', marginBottom: '1.25rem' }}>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
          <h1 style={{fontSize: '2rem'}}>CareerShield Intelligence</h1>
        </div>
        <p className="subtitle" style={{ maxWidth: '600px', margin: '0 auto', fontSize: '0.9rem', marginBottom: '0' }}>
          MLOps-powered early warning system for company risk forecasting and precision career transition advisory.
        </p>
      </div>

      <div className="grid-2">
        <div>
          <PredictForm onPredict={handlePredict} isLoading={isLoading} />
          {error && (
            <div className="glass-panel animate-in" style={{ marginTop: '1.5rem', borderLeft: '4px solid var(--risk-high)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <AlertTriangle color="var(--risk-high)" />
              <p style={{ color: 'var(--risk-high)', margin: 0, fontWeight: 500 }}>{error}</p>
            </div>
          )}
        </div>

        <div>
          {result ? (
            <ResultCard result={result} />
          ) : !isLoading ? (
            <div className="glass-panel animate-in" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '300px', textAlign: 'center', opacity: 0.6 }}>
              <p>Submit company metrics to generate a real-time risk profile.</p>
            </div>
          ) : (
            <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '300px' }}>
              <div className="loader" style={{ width: '40px', height: '40px', borderWidth: '4px', marginBottom: '1rem' }}></div>
              <p>Analyzing historical data patterns & model inference...</p>
            </div>
          )}
        </div>
      </div>
      
      {result && (
         <div style={{ marginTop: '2rem' }}>
           <CareerPath careerData={result.career_advice} />
         </div>
      )}
    </div>
  );
}

export default App;
