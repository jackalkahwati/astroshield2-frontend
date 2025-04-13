import React, { useState } from 'react';
import Head from 'next/head';
import SatelliteAnalysis from '../components/ccdm/SatelliteAnalysis';
import ThreatAssessment from '../components/ccdm/ThreatAssessment';
import HistoricalAnalysis from '../components/ccdm/HistoricalAnalysis';
import ShapeChangeDetection from '../components/ccdm/ShapeChangeDetection';

// Sample satellites for the dropdown
const SAMPLE_SATELLITES = [
  { noradId: 25544, name: "ISS (International Space Station)" },
  { noradId: 48274, name: "Starlink-1234" },
  { noradId: 43013, name: "NOAA-20" },
  { noradId: 33591, name: "Hubble Space Telescope" },
  { noradId: 27424, name: "XMM-Newton" }
];

export default function CCDMDashboard() {
  const [selectedNoradId, setSelectedNoradId] = useState(null);
  const [activeTab, setActiveTab] = useState('threat');
  const [customDateRange, setCustomDateRange] = useState(null);

  const handleSatelliteChange = (e) => {
    const noradId = parseInt(e.target.value, 10);
    setSelectedNoradId(noradId || null);
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <Head>
        <title>CCDM Dashboard | AstroShield</title>
        <meta name="description" content="Concealment, Camouflage, Deception, and Maneuvering Dashboard" />
      </Head>
      
      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-800">CCDM Dashboard</h1>
          <p className="text-gray-600 mt-2">
            Concealment, Camouflage, Deception, and Maneuvering Analysis
          </p>
        </div>

        {/* Satellite Selection and Tabs */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
            <div className="w-full sm:w-auto">
              <label htmlFor="satellite-select" className="block text-sm font-medium text-gray-700 mb-1">
                Select Satellite
              </label>
              <select
                id="satellite-select"
                className="block w-full sm:w-64 p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                onChange={handleSatelliteChange}
                value={selectedNoradId || ''}
              >
                <option value="">-- Select a satellite --</option>
                {SAMPLE_SATELLITES.map((satellite) => (
                  <option key={satellite.noradId} value={satellite.noradId}>
                    {satellite.name} (NORAD ID: {satellite.noradId})
                  </option>
                ))}
              </select>
            </div>
            
            {selectedNoradId && (
              <div className="w-full sm:w-auto flex overflow-x-auto sm:overflow-visible">
                <nav className="flex space-x-4">
                  {[
                    { id: 'threat', label: 'Threat Assessment' },
                    { id: 'analysis', label: 'Satellite Analysis' },
                    { id: 'historical', label: 'Historical Data' },
                    { id: 'shape', label: 'Shape Changes' }
                  ].map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => handleTabChange(tab.id)}
                      className={`px-3 py-2 text-sm font-medium rounded-md whitespace-nowrap ${
                        activeTab === tab.id
                          ? 'bg-blue-100 text-blue-700'
                          : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </nav>
              </div>
            )}
          </div>
        </div>

        {/* Content Area */}
        {selectedNoradId ? (
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            {/* Threat Assessment View */}
            {activeTab === 'threat' && (
              <ThreatAssessment noradId={selectedNoradId} />
            )}
            
            {/* Satellite Analysis View */}
            {activeTab === 'analysis' && (
              <SatelliteAnalysis noradId={selectedNoradId} />
            )}
            
            {/* Historical Analysis View */}
            {activeTab === 'historical' && (
              <div className="p-6">
                <div className="mb-4">
                  <div className="flex justify-between items-center">
                    <h2 className="text-2xl font-bold text-gray-800">Historical Analysis</h2>
                    <div className="flex space-x-2">
                      <button
                        onClick={() => setCustomDateRange(null)}
                        className={`px-3 py-1 text-sm rounded-md ${
                          !customDateRange
                            ? 'bg-blue-100 text-blue-700'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}
                      >
                        Last 7 Days
                      </button>
                      <button
                        onClick={() => {
                          const endDate = new Date();
                          const startDate = new Date();
                          startDate.setDate(endDate.getDate() - 30);
                          setCustomDateRange({ startDate, endDate });
                        }}
                        className={`px-3 py-1 text-sm rounded-md ${
                          customDateRange
                            ? 'bg-blue-100 text-blue-700'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}
                      >
                        Last 30 Days
                      </button>
                    </div>
                  </div>
                </div>
                <HistoricalAnalysis
                  noradId={selectedNoradId}
                  customDateRange={customDateRange}
                />
              </div>
            )}
            
            {/* Shape Change View */}
            {activeTab === 'shape' && (
              <ShapeChangeDetection noradId={selectedNoradId} />
            )}
          </div>
        ) : (
          <div className="bg-blue-50 border border-blue-200 text-blue-700 px-6 py-12 rounded-md text-center">
            <svg
              className="mx-auto h-12 w-12 text-blue-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <h2 className="mt-2 text-lg font-medium text-blue-800">Select a Satellite</h2>
            <p className="mt-1 text-sm text-blue-600">
              Please select a satellite from the dropdown menu to view CCDM analysis.
            </p>
          </div>
        )}
      </main>

      <footer className="bg-white border-t border-gray-200 py-6 mt-8">
        <div className="container mx-auto px-4">
          <p className="text-center text-gray-500 text-sm">
            &copy; {new Date().getFullYear()} AstroShield. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
} 