/**
 * Bengaluru Road Accident Risk Prediction and Safety Recommendation System
 * Script: static/js/map.js
 * 
 * Interactive Leaflet.js Mapping Engine for Bengaluru:
 * - Centered on Bengaluru (12.9716° N, 77.5946° E)
 * - Official BTP Accident Blackspots Layer
 * - BTP Traffic Police Stations Layer
 * - Dynamic Route Polyline Drawing with Checkpoint Risk Badges
 */

class BengaluruMapManager {
    constructor(mapContainerId, initialLat = 12.9716, initialLon = 77.5946, zoom = 12) {
        this.containerId = mapContainerId;
        this.center = [initialLat, initialLon];
        this.zoom = zoom;
        this.map = null;
        this.blackspotLayer = null;
        this.stationLayer = null;
        this.routeLayer = null;
        this.markerLayer = null;
        this.initMap();
    }

    initMap() {
        const el = document.getElementById(this.containerId);
        if (!el) return;

        // Create Leaflet map
        this.map = L.map(this.containerId).setView(this.center, this.zoom);

        // OpenStreetMap Tile Layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(this.map);

        this.blackspotLayer = L.layerGroup().addTo(this.map);
        this.stationLayer = L.layerGroup().addTo(this.map);
        this.routeLayer = L.layerGroup().addTo(this.map);
        this.markerLayer = L.layerGroup().addTo(this.map);

        // Layer controls
        const overlays = {
            "BTP Accident Blackspots": this.blackspotLayer,
            "Traffic Police Stations": this.stationLayer,
            "Active Driving Route": this.routeLayer
        };
        L.control.layers(null, overlays, { collapsed: false, position: 'topright' }).addTo(this.map);
    }

    loadBlackspots() {
        fetch('/api/blackspots')
            .then(res => res.json())
            .then(geojson => {
                this.blackspotLayer.clearLayers();
                L.geoJSON(geojson, {
                    pointToLayer: (feature, latlng) => {
                        const cat = feature.properties.risk_category || 'HIGH';
                        const color = cat === 'CRITICAL' ? '#dc2626' : (cat === 'HIGH' ? '#ea580c' : '#f59e0b');
                        const radius = cat === 'CRITICAL' ? 10 : (cat === 'HIGH' ? 8 : 6);

                        return L.circleMarker(latlng, {
                            radius: radius,
                            fillColor: color,
                            color: '#ffffff',
                            weight: 2,
                            opacity: 1,
                            fillOpacity: 0.85
                        });
                    },
                    onEachFeature: (feature, layer) => {
                        const p = feature.properties;
                        const popupHtml = `
                            <div class="p-1">
                                <h6 class="fw-bold mb-1 text-danger"><i class="fa-solid fa-triangle-exclamation me-1"></i>${p.name}</h6>
                                <p class="mb-1 fs-7"><strong>Risk Level:</strong> <span class="badge ${p.risk_category === 'CRITICAL' ? 'bg-danger' : 'bg-warning'}">${p.risk_category}</span></p>
                                <p class="mb-1 fs-7"><strong>Historical 3-Yr Accidents:</strong> ${p.accidents_3yr}</p>
                                <p class="mb-1 fs-7"><strong>Corridor:</strong> ${p.corridor}</p>
                                <p class="mb-0 fs-8 text-muted"><strong>Hazard Factors:</strong> ${p.hazards}</p>
                            </div>
                        `;
                        layer.bindPopup(popupHtml);
                    }
                }).addTo(this.blackspotLayer);
            })
            .catch(err => console.error('Error loading blackspots:', err));
    }

    loadBTPStations() {
        fetch('/api/btp-stations')
            .then(res => res.json())
            .then(geojson => {
                this.stationLayer.clearLayers();
                L.geoJSON(geojson, {
                    pointToLayer: (feature, latlng) => {
                        return L.circleMarker(latlng, {
                            radius: 5,
                            fillColor: '#3b82f6',
                            color: '#ffffff',
                            weight: 1.5,
                            opacity: 1,
                            fillOpacity: 0.75
                        });
                    },
                    onEachFeature: (feature, layer) => {
                        const p = feature.properties;
                        const popupHtml = `
                            <div class="p-1">
                                <h6 class="fw-bold mb-1 text-primary"><i class="fa-solid fa-shield-halved me-1"></i>${p.station}</h6>
                                <p class="mb-1 fs-7"><strong>Zone:</strong> ${p.zone}</p>
                                <p class="mb-1 fs-7"><strong>2024 Total Crashes:</strong> ${p.total_crashes}</p>
                                <p class="mb-0 fs-7 text-danger"><strong>Fatal Crashes:</strong> ${p.fatal_crashes}</p>
                            </div>
                        `;
                        layer.bindPopup(popupHtml);
                    }
                }).addTo(this.stationLayer);
            })
            .catch(err => console.error('Error loading BTP stations:', err));
    }

    renderRoute(routeData) {
        this.routeLayer.clearLayers();
        this.markerLayer.clearLayers();

        const waypoints = routeData.route.waypoints;
        if (!waypoints || waypoints.length === 0) return;

        // Choose polyline color based on risk score
        const score = routeData.overall_risk.risk_score;
        const color = score >= 70 ? '#dc2626' : (score >= 45 ? '#ea580c' : '#10b981');

        // Draw polyline
        const polyline = L.polyline(waypoints, {
            color: color,
            weight: 6,
            opacity: 0.85,
            smoothFactor: 1
        }).addTo(this.routeLayer);

        // Add Origin Marker (Green Pin)
        const orig = routeData.origin;
        const originMarker = L.marker([orig.lat, orig.lon], {
            icon: L.divIcon({
                className: 'custom-pin',
                html: '<div style="background-color:#10b981; color:white; border-radius:50%; width:28px; height:28px; display:flex; align-items:center; justify-content:center; border:2px solid white; box-shadow:0 2px 4px rgba(0,0,0,0.3);"><i class="fa-solid fa-location-dot"></i></div>',
                iconSize: [28, 28],
                iconAnchor: [14, 28]
            })
        }).bindPopup(`<strong>Origin:</strong> ${orig.name}`).addTo(this.markerLayer);

        // Add Destination Marker (Red Flag)
        const dest = routeData.destination;
        const destMarker = L.marker([dest.lat, dest.lon], {
            icon: L.divIcon({
                className: 'custom-pin',
                html: '<div style="background-color:#ef4444; color:white; border-radius:50%; width:28px; height:28px; display:flex; align-items:center; justify-content:center; border:2px solid white; box-shadow:0 2px 4px rgba(0,0,0,0.3);"><i class="fa-solid fa-flag-checkered"></i></div>',
                iconSize: [28, 28],
                iconAnchor: [14, 28]
            })
        }).bindPopup(`<strong>Destination:</strong> ${dest.name}`).addTo(this.markerLayer);

        // Add Checkpoint risk markers
        if (routeData.checkpoint_samples) {
            routeData.checkpoint_samples.forEach(cp => {
                const cpColor = cp.risk_score >= 70 ? '#dc2626' : (cp.risk_score >= 45 ? '#ea580c' : '#10b981');
                L.circleMarker([cp.lat, cp.lon], {
                    radius: 4,
                    fillColor: cpColor,
                    color: '#ffffff',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.9
                }).bindPopup(`Checkpoint #${cp.checkpoint_id}<br><strong>Risk:</strong> ${cp.risk_level} (${cp.risk_score}/100)`).addTo(this.routeLayer);
            });
        }

        // Fit map bounds to polyline
        this.map.fitBounds(polyline.getBounds(), { padding: [40, 40] });
    }
}

window.BengaluruMapManager = BengaluruMapManager;
