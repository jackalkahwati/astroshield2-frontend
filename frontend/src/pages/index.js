import React from 'react';
import Head from 'next/head';
import StatusPanel from '../components/StatusPanel';

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <Head>
        <title>AstroShield Platform</title>
        <meta name="description" content="Space Situational Awareness & Satellite Protection System" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="container mx-auto px-4 py-10">
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-6xl font-bold mb-4 text-blue-400">AstroShield Platform</h1>
          <p className="text-xl text-gray-300">Space Situational Awareness & Satellite Protection System</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          <div className="md:col-span-2">
            <div className="bg-gray-800 p-8 rounded-lg shadow-lg">
              <h2 className="text-2xl font-bold mb-6">System Status</h2>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <div className="bg-gray-700 p-6 rounded-lg">
                  <h3 className="text-xl font-bold mb-4 text-blue-300">Frontend Server</h3>
                  <div className="flex items-center">
                    <span className="w-3 h-3 bg-green-500 rounded-full mr-2"></span>
                    <span>ONLINE</span>
                  </div>
                </div>
                
                <div className="bg-gray-700 p-6 rounded-lg">
                  <h3 className="text-xl font-bold mb-4 text-blue-300">HTTPS Encryption</h3>
                  <div className="flex items-center">
                    <span className="w-3 h-3 bg-green-500 rounded-full mr-2"></span>
                    <span>ACTIVE</span>
                  </div>
                </div>
                
                <div className="bg-gray-700 p-6 rounded-lg">
                  <h3 className="text-xl font-bold mb-4 text-blue-300">API Backend</h3>
                  <div className="flex items-center">
                    <span className="w-3 h-3 bg-yellow-500 rounded-full mr-2"></span>
                    <span>IN PROGRESS</span>
                  </div>
                </div>
                
                <div className="bg-gray-700 p-6 rounded-lg">
                  <h3 className="text-xl font-bold mb-4 text-blue-300">Database Connection</h3>
                  <div id="database-status" className="flex items-center">
                    <span id="database-status-indicator" className="w-3 h-3 bg-yellow-500 rounded-full mr-2"></span>
                    <span>PENDING</span>
                  </div>
                </div>
              </div>
              
              <div className="mt-6 text-gray-400">
                <p>The deployment is in progress. The frontend is accessible, and we're currently configuring the backend services.</p>
              </div>
            </div>
          </div>
          
          <div>
            <StatusPanel />
            
            <div className="bg-gray-800 p-6 rounded-lg shadow-lg mt-8">
              <h2 className="text-xl font-bold mb-4">Quick Actions</h2>
              <ul className="space-y-3">
                <li>
                  <a href="/api/v1/docs" className="text-blue-400 hover:text-blue-300 flex items-center">
                    <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                    </svg>
                    API Documentation
                  </a>
                </li>
                <li>
                  <a href="/api/v1/health" className="text-blue-400 hover:text-blue-300 flex items-center">
                    <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                      <path fillRule="evenodd" d="M10 2a8 8 0 100 16 8 8 0 000-16zm0 14a6 6 0 100-12 6 6 0 000 12z" clipRule="evenodd" />
                    </svg>
                    Health Check
                  </a>
                </li>
                <li>
                  <a href="/dashboard" className="text-blue-400 hover:text-blue-300 flex items-center">
                    <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                      <path d="M5 3a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2V5a2 2 0 00-2-2H5zM5 11a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2v-2a2 2 0 00-2-2H5zM11 5a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V5zM11 13a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                    </svg>
                    Dashboard
                  </a>
                </li>
                <li>
                  <a href="/ccdm" className="text-blue-400 hover:text-blue-300 flex items-center">
                    <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                    </svg>
                    CCDM Dashboard
                  </a>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </main>

      <footer className="container mx-auto px-4 py-8 mt-12 border-t border-gray-800">
        <div className="text-center text-gray-500">
          <p>© 2024 AstroShield. All rights reserved.</p>
          <p className="mt-2 text-sm">Space Situational Awareness & Satellite Protection System</p>
        </div>
      </footer>
      
      <script dangerouslySetInnerHTML={{
        __html: `
          // JavaScript to update the database status based on the API response
          document.addEventListener('DOMContentLoaded', function() {
            fetch('/api/v1/health')
              .then(response => response.json())
              .then(data => {
                if (data.services.database === 'connected') {
                  document.getElementById('database-status-indicator').className = 'w-3 h-3 bg-green-500 rounded-full mr-2';
                  document.getElementById('database-status').querySelector('span:last-child').textContent = 'CONNECTED';
                }
              })
              .catch(error => {
                console.error('Error fetching health status:', error);
              });
          });
        `
      }} />
    </div>
  );
} 