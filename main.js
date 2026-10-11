/**
 * Bengaluru Road Accident Risk Prediction and Safety Recommendation System
 * Script: static/js/main.js
 * 
 * Dashboard Charts & Dynamic Statistics Engine
 */

document.addEventListener('DOMContentLoaded', () => {
    // If dashboard charts container is present, load stats
    if (document.getElementById('monthlyChart')) {
        loadDashboardCharts();
    }
});

function loadDashboardCharts() {
    fetch('/api/stats')
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                console.error(data.error);
                return;
            }

            // 1. Monthly Trends Chart
            const ctxMonthly = document.getElementById('monthlyChart').getContext('2d');
            new Chart(ctxMonthly, {
                type: 'line',
                data: {
                    labels: data.monthly_labels,
                    datasets: [{
                        label: 'Total Incident / Accident Reports',
                        data: data.monthly_data,
                        borderColor: '#2563eb',
                        backgroundColor: 'rgba(37, 99, 235, 0.1)',
                        fill: true,
                        tension: 0.35,
                        pointBackgroundColor: '#1e40af',
                        pointRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: false, grid: { color: '#f1f5f9' } },
                        x: { grid: { display: false } }
                    }
                }
            });

            // 2. Day of the Week Chart
            const ctxDaily = document.getElementById('dailyChart').getContext('2d');
            new Chart(ctxDaily, {
                type: 'bar',
                data: {
                    labels: data.daily_labels,
                    datasets: [{
                        label: 'Accident Reports',
                        data: data.daily_data,
                        backgroundColor: data.daily_labels.map(d => (d === 'Saturday' || d === 'Sunday' || d === 'Friday') ? '#dc2626' : '#3b82f6'),
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true, grid: { color: '#f1f5f9' } },
                        x: { grid: { display: false } }
                    }
                }
            });

            // 3. High-Risk Corridors Chart
            const ctxCorridor = document.getElementById('corridorChart').getContext('2d');
            new Chart(ctxCorridor, {
                type: 'bar',
                data: {
                    labels: data.corridor_labels,
                    datasets: [{
                        axis: 'y',
                        label: 'Recorded Incidents',
                        data: data.corridor_data,
                        backgroundColor: '#ea580c',
                        borderRadius: 4
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { beginAtZero: true, grid: { color: '#f1f5f9' } },
                        y: { grid: { display: false } }
                    }
                }
            });

            // 4. Weather Impact Chart
            const ctxWeather = document.getElementById('weatherChart').getContext('2d');
            const wLabels = Object.keys(data.weather_impact);
            const wValues = Object.values(data.weather_impact);
            new Chart(ctxWeather, {
                type: 'bar',
                data: {
                    labels: wLabels,
                    datasets: [{
                        label: 'Avg Incidents / Day',
                        data: wValues,
                        backgroundColor: ['#10b981', '#3b82f6', '#f59e0b', '#dc2626', '#8b5cf6'],
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true, grid: { color: '#f1f5f9' } },
                        x: { grid: { display: false } }
                    }
                }
            });

            // 5. Risk Tiers Doughnut Chart
            const ctxTier = document.getElementById('tierChart').getContext('2d');
            const tLabels = Object.keys(data.risk_tiers);
            const tValues = Object.values(data.risk_tiers);
            new Chart(ctxTier, {
                type: 'doughnut',
                data: {
                    labels: tLabels,
                    datasets: [{
                        data: tValues,
                        backgroundColor: ['#f59e0b', '#10b981', '#dc2626'],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom' }
                    }
                }
            });
        })
        .catch(err => console.error('Failed to load dashboard statistics:', err));
}
