// Initialize violation chart
const violationsCtx = document.getElementById('violationsChart').getContext('2d');
const violationsChart = new Chart(violationsCtx, {
    type: 'bar',
    data: {
        labels: ['Kinematic', 'RF Power', 'Proximity'],
        datasets: [{
            label: 'Violations in Last Hour',
            data: [0, 0, 0],
            backgroundColor: [
                'rgba(255, 99, 132, 0.5)',
                'rgba(54, 162, 235, 0.5)',
                'rgba(255, 206, 86, 0.5)'
            ]
        }]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});

// Initialize maneuvers chart
const maneuversCtx = document.getElementById('maneuversChart').getContext('2d');
const maneuversChart = new Chart(maneuversCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Maneuver Confidence',
            data: [],
            borderColor: 'rgba(75, 192, 192, 1)',
            tension: 0.1
        }]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                max: 1
            }
        }
    }
});

// Initialize CCDM chart
const ccdmCtx = document.getElementById('ccdmChart').getContext('2d');
const ccdmChart = new Chart(ccdmCtx, {
    type: 'bubble',
    data: {
        datasets: [{
            label: 'High Risk Conjunctions',
            data: [],
            backgroundColor: 'rgba(255, 99, 132, 0.7)',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 1
        }, {
            label: 'Low Risk Conjunctions',
            data: [],
            backgroundColor: 'rgba(75, 192, 192, 0.7)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1
        }]
    },
    options: {
        responsive: true,
        scales: {
            x: {
                type: 'linear',
                position: 'bottom',
                title: {
                    display: true,
                    text: 'Miss Distance (km)'
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'Probability of Collision'
                }
            }
        }
    }
});

// Update charts periodically
async function updateCharts() {
    try {
        // Fetch health and trends data
        const healthResponse = await fetch('/api/health');
        let trendsData = null;
        try {
            // Try to get trends for a default spacecraft (ID: 1)
            const trendsResponse = await fetch('/api/spacecraft/1/ccdm/trends?hours=24', {
                headers: {
                    'Accept': 'application/json',
                    'Cache-Control': 'no-cache'
                }
            });
            let defaultData = {
                total_conjunctions: 0,
                risk_levels: { critical: 0, high: 0, moderate: 0, low: 0 },
                temporal_metrics: {
                    hourly_rate: 0,
                    peak_hour: null,
                    trend_direction: 'stable'
                },
                velocity_metrics: {
                    average_velocity: 0,
                    max_velocity: 0,
                    velocity_trend: 'stable'
                }
            };
            
            if (!trendsResponse.ok) {
                console.error(`HTTP error! status: ${trendsResponse.status}`);
                return defaultData;
            }
            
            try {
                const data = await trendsResponse.json();
                if (!data || data.error) {
                    console.warn('Trends data error:', data?.error);
                    return defaultData;
                }
                return data;
            } catch (error) {
                console.error('Error parsing trends response:', error);
                return defaultData;
            }
        } catch (error) {
            console.error('Error fetching trends data:', error);
            return defaultData;
        }

        const healthData = healthResponse.ok ? await healthResponse.json() : null;
        
        if (!healthData) {
            console.error('Failed to fetch health data');
        }

        // Update health status
        const healthStatus = document.getElementById('healthStatus');
        if (healthStatus) {
            if (healthData.status === 'healthy') {
                healthStatus.className = 'alert alert-success';
                healthStatus.textContent = 'All systems operational';
            } else {
                healthStatus.className = 'alert alert-danger';
                healthStatus.textContent = 'System issues detected';
            }
        }

        // Update trends information
        if (trendsData && !trendsData.error) {
            const summaryDiv = document.getElementById('ccdmSummary');
            if (summaryDiv) {
                const risk_levels = trendsData.risk_levels || {
                    critical: 0,
                    high: 0,
                    moderate: 0,
                    low: 0
                };
                const temporal_metrics = trendsData.temporal_metrics || {
                    hourly_rate: 0,
                    trend_direction: 'stable'
                };
                const velocity_metrics = trendsData.velocity_metrics || {
                    average_velocity: 0,
                    max_velocity: 0,
                    velocity_trend: 'stable'
                };
                
                summaryDiv.innerHTML = `
                    <div class="row">
                        <div class="col-md-3">
                            <h5>Risk Levels</h5>
                            <ul class="list-unstyled">
                                <li>Critical: ${risk_levels.critical}</li>
                                <li>High: ${risk_levels.high}</li>
                                <li>Moderate: ${risk_levels.moderate}</li>
                                <li>Low: ${risk_levels.low}</li>
                            </ul>
                        </div>
                        <div class="col-md-3">
                            <h5>Temporal Metrics</h5>
                            <ul class="list-unstyled">
                                <li>Hourly Rate: ${temporal_metrics.hourly_rate}</li>
                                <li>Trend: ${temporal_metrics.trend_direction}</li>
                            </ul>
                        </div>
                        <div class="col-md-3">
                            <h5>Velocity Metrics</h5>
                            <ul class="list-unstyled">
                                <li>Average: ${velocity_metrics.average_velocity} m/s</li>
                                <li>Max: ${velocity_metrics.max_velocity} m/s</li>
                                <li>Trend: ${velocity_metrics.velocity_trend}</li>
                            </ul>
                        </div>
                    </div>
                `;
            }
        }
    } catch (error) {
        console.error('Error updating health and trends:', error);
        const healthStatus = document.getElementById('healthStatus');
        if (healthStatus) {
            healthStatus.className = 'alert alert-danger';
            healthStatus.textContent = 'Error updating system status';
        }
    }
}

// Update CCDM data
async function fetchTrendsData() {
    try {
        let latestData = null;
        let historicalData = null;
        
        try {
            const latestResponse = await fetch('/api/spacecraft/ccdm/latest');
            if (latestResponse.ok) {
                latestData = await latestResponse.json();
            }
        } catch (error) {
            console.error('Error fetching latest CCDM data:', error);
        }
        
        try {
            const historicalResponse = await fetch('/api/spacecraft/ccdm/historical');
            if (historicalResponse.ok) {
                historicalData = await historicalResponse.json();
            }
        } catch (error) {
            console.error('Error fetching historical CCDM data:', error);
        }

        if (!latestData && !historicalData) {
            throw new Error('Failed to fetch any CCDM data');
        }

            // Process indicators
            const highRiskData = [];
            const lowRiskData = [];
            const latestIndicators = latestData.indicators || [];
            const historicalIndicators = historicalData.indicators || [];
            const allIndicators = [...latestIndicators, ...historicalIndicators];

            allIndicators.forEach(indicator => {
                if (!indicator.miss_distance || !indicator.probability_of_collision) {
                    return;
                }
                
                const dataPoint = {
                    x: indicator.miss_distance / 1000, // Convert to km
                    y: indicator.probability_of_collision,
                    r: Math.max(5, Math.min(20, (indicator.relative_velocity || 0) / 50)) // Bubble size based on relative velocity
                };

                if (indicator.probability_of_collision !== null && indicator.probability_of_collision !== undefined) {
                    if (indicator.probability_of_collision > 0.5) {
                        highRiskData.push(dataPoint);
                    } else {
                        lowRiskData.push(dataPoint);
                    }
                }
            });

            // Update chart data
            ccdmChart.data.datasets[0].data = highRiskData;
            ccdmChart.data.datasets[1].data = lowRiskData;
            ccdmChart.update();

            // Update summary information
            const summaryDiv = document.getElementById('ccdmSummary');
            if (summaryDiv && historicalData.summary) {
                const summary = historicalData.summary;
                summaryDiv.innerHTML = `
                    <div class="row">
                        <div class="col-md-4">
                            <h5>Total Conjunctions</h5>
                            <p>${summary.total_conjunctions || 0}</p>
                        </div>
                        <div class="col-md-4">
                            <h5>High Risk Events</h5>
                            <p>${summary.high_risk_conjunctions || 0}</p>
                        </div>
                        <div class="col-md-4">
                            <h5>Avg Miss Distance</h5>
                            <p>${((summary.average_miss_distance || 0) / 1000).toFixed(2)} km</p>
                        </div>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Error updating CCDM data:', error);
            const summaryDiv = document.getElementById('ccdmSummary');
            if (summaryDiv) {
                summaryDiv.innerHTML = `
                    <div class="alert alert-danger">
                        Failed to load CCDM data. Please try again later.
                    </div>
                `;
            }
        }
    }
}

// Initial fetch of trends data
fetchTrendsData();

// Initial update
updateCharts()
    .catch(error => console.error('Error updating charts:', error));

// Update every 30 seconds
setInterval(function() {
    updateCharts()
        .catch(error => console.error('Error updating charts:', error));
}, 30000);
