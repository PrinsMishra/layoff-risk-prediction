import React, { useState, useEffect } from 'react';
// Pipeline Trigger Test 2
import { Search } from 'lucide-react';
import { fetchMetadataLists } from '../api/predict';

const PredictForm = ({ onPredict, isLoading }) => {
  const [formData, setFormData] = useState({
    industry: 'Software',
    department: 'Engineering',
    ai_exposure: 'Partial',
    total_employees: 5000
  });

  const [metadata, setMetadata] = useState({ industries: [], departments: [] });

  useEffect(() => {
    fetchMetadataLists().then(setMetadata);
  }, []);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onPredict({
      ...formData,
      total_employees: parseInt(formData.total_employees, 10) || 0
    });
  };

  return (
    <div className="glass-panel animate-in" style={{ animationDelay: '0.1s' }}>
      <h2>Layoff Risk Analysis</h2>
      <p className="subtitle">Enter company metrics to evaluate vulnerability patterns.</p>

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Industry Sector</label>
          <input
            type="text"
            list="industry-list"
            name="industry"
            value={formData.industry}
            onChange={handleChange}
            className="form-input"
            placeholder="Search or type industry..."
            required
            autoComplete="off"
          />
          <datalist id="industry-list">
            {metadata.industries.map((ind, i) => (
              <option key={i} value={ind} />
            ))}
          </datalist>
        </div>

        <div className="form-group">
          <label>Primary Department</label>
          <input
            type="text"
            list="dept-list"
            name="department"
            value={formData.department}
            onChange={handleChange}
            className="form-input"
            placeholder="Search or type department..."
            required
            autoComplete="off"
          />
          <datalist id="dept-list">
            {metadata.departments.map((dept, i) => (
              <option key={i} value={dept} />
            ))}
          </datalist>
        </div>

        <div className="grid-2" style={{ gap: '1rem', marginBottom: '1.25rem' }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label>AI Exposure</label>
            <select
              name="ai_exposure"
              value={formData.ai_exposure}
              onChange={handleChange}
              className="form-input"
            >
              <option value="No">No / Low</option>
              <option value="Partial">Partial</option>
              <option value="Yes">Yes / High</option>
            </select>
          </div>

          <div className="form-group" style={{ marginBottom: 0 }}>
            <label>Company Size (Employees)</label>
            <input
              type="number"
              name="total_employees"
              value={formData.total_employees}
              onChange={handleChange}
              className="form-input"
              min="1"
              required
            />
          </div>
        </div>

        <button type="submit" className="btn-primary" disabled={isLoading}>
          {isLoading ? <><span className="loader" style={{marginRight: '8px'}}></span> Analyzing...</> : <><Search size={20} /> Analyze Risk</>}
        </button>
      </form>
    </div>
  );
};

export default PredictForm;
