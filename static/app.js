// Firebase Configuration
const firebaseConfig = {
    apiKey: "AIzaSyBsiQZIwtAyQcBNJMUWziJ6mwCTCV3SWCk",
    authDomain: "agri-hill.firebaseapp.com",
    databaseURL: "https://agri-hill-default-rtdb.asia-southeast1.firebasedatabase.app",
    projectId: "agri-hill",
    storageBucket: "agri-hill.firebasestorage.app",
    messagingSenderId: "234302354938",
    appId: "1:234302354938:web:a9eae33ea5d1e7daed2043"
};

// Initialize Firebase
let app;
let database;
try {
    app = firebase.initializeApp(firebaseConfig);
    database = firebase.database();
    console.log('Firebase initialized successfully');
} catch (error) {
    console.error('Error initializing Firebase:', error);
    showToast('Firebase Error', 'Failed to initialize Firebase: ' + error.message, 'error');
}

// Global variables
const apiKey = 'f4ccd28ceb577a556911dfd947e0ab84';
let sensorData = {
    temperature: 0,
    humidity: 0,
    soilMoisture: 0,
    soilMoisturePercent: 0,
    waterLevel: 0,
    waterLevelPercent: 0,
    pump: false
};

let cropData = {
    cropId: 0,
    soilType: 0,
    seedlingStage: 0,
    currentMoisture: 0,
    threshold: 0
};

let rainForecast = { chance: 0, amount: 0 };
let manualOverride = false;

// Crop data
const cropMoistureRequirements = {
    0: { base: 35, name: 'Ginger' },
    1: { base: 40, name: 'Large Cardamom' },
    2: { base: 28, name: 'Maize (corn)' },
    3: { base: 35, name: 'Paddy (rice)' },
    4: { base: 34, name: 'Turmeric' },
    5: { base: 40, name: 'Bananas' },
    6: { base: 30, name: 'Beans' },
    7: { base: 32, name: 'Cabbage' },
    8: { base: 30, name: 'Kiwi' },
    9: { base: 28, name: 'Mustard' },
    10:{ base: 30, name: 'Orange' },
    11:{ base: 32, name: 'Papaya' },
    12:{ base: 30, name: 'Peas' },
    13:{ base: 45, name: 'Potato' },
    14:{ base: 35, name: 'Radish' },
    15:{ base: 28, name: 'Soybeans' },
    16:{ base: 42, name: 'Sweet Potato' },
    17:{ base: 30, name: 'Tomato' }
};

const soilMoistureFactors = { 0: 1.2, 1: 1.3, 2: 0.8, 3: 1.4, 4: 1.1, 5: 0.9, 6: 0.7 };
const stageMoistureMultipliers = { 0: 1.2, 1: 1.3, 2: 1.0, 3: 0.8, 4: 1.1 };

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    if (window.lucide) lucide.createIcons();
    
    // Check if Firebase is loaded
    if (typeof firebase === 'undefined') {
        console.error('Firebase SDK not loaded');
        showToast('Error', 'Firebase SDK failed to load', 'error');
        updateConnectionStatus('disconnected');
        return;
    }
    
    // Initialize Firebase and features
    initializeFirebase();
    initializeFirebaseListeners();
    calculateThreshold();
    updateRecommendation();
    
    // Set up Firebase connection monitoring
    const connectedRef = database.ref('.info/connected');
    connectedRef.on('value', (snap) => {
        if (snap.val() === true) {
            console.log('Connected to Firebase');
            updateConnectionStatus('connected');
            showToast('Connected', 'Successfully connected to Firebase database', 'success');
        } else {
            console.log('Disconnected from Firebase');
            updateConnectionStatus('disconnected');
            showToast('Disconnected', 'Lost connection to Firebase database', 'error');
        }
    }, (error) => {
        console.error('Error monitoring connection:', error);
        updateConnectionStatus('disconnected');
        showToast('Connection Error', 'Failed to monitor Firebase connection: ' + error.message, 'error');
    });
});

// Firebase functions
function initializeFirebase() {
    updateConnectionStatus('connecting');
}

function initializeFirebaseListeners() {
    if (!database) {
        console.error('Firebase database not initialized');
        updateConnectionStatus('disconnected');
        return;
    }

    // Listen for sensor data changes
    const sensorsRef = database.ref('sensors');
    sensorsRef.on('value', (snapshot) => {
        const data = snapshot.val();
        console.log('Received sensor data:', data);
        if (data) {
            // Get all sensor values from Firebase
            sensorData = {
                ...sensorData,
                temperature: data.temperature ?? 0,
                humidity: data.humidity ?? 0,
                soilMoisture: data.soilMoisture ?? 0,
                soilMoisturePercent: data.soilMoisturePercent ?? 0,
                waterLevel: data.waterLevel ?? 0,
                waterLevelPercent: data.waterLevelPercent ?? 0
            };
            
            // Update the current moisture input with soil moisture percentage from Firebase
            const currentMoistureInput = document.getElementById('current-moisture');
            if (currentMoistureInput) {
                currentMoistureInput.value = data.soilMoisturePercent ?? 0;
                // Trigger the input event to update displays
                currentMoistureInput.dispatchEvent(new Event('input'));
            }
            updateSensorDisplay();
            calculateThreshold();
            updateConnectionStatus('connected');
        }
    }, (error) => {
        console.error('Error fetching sensor data:', error);
        updateConnectionStatus('disconnected');
    });

    // Listen for actuator states
    const actuatorsRef = database.ref('actuators');
    actuatorsRef.on('value', (snapshot) => {
        const data = snapshot.val();
        if (data) {
            // Update pump status
            if (data.pump !== undefined) {
                sensorData.pump = data.pump;
                updateSensorDisplay();
            }
            
            // Update manual override status
            if (data.manualOverride !== undefined) {
                manualOverride = data.manualOverride;
                document.getElementById('manual-override').checked = data.manualOverride;
                document.getElementById('mode-display').textContent = data.manualOverride ? 'Manual' : 'Automatic';
                document.getElementById('pump-on-btn').disabled = !data.manualOverride;
                document.getElementById('pump-off-btn').disabled = !data.manualOverride;
            }
        }
    });
}

// Weather functionality
async function fetchWeather() {
    const location = document.getElementById('location').value.trim();
    if (!location) { showToast('Location Required', 'Please enter a city name', 'error'); return; }
    const btn = document.getElementById('weather-btn');
    btn.textContent = 'Loading...'; btn.disabled = true;
    try {
        const response = await fetch(`/weather?location=${encodeURIComponent(location)}`);
        if (!response.ok) {
            console.error('HTTP Error:', response.status, response.statusText);
            throw new Error(`HTTP Error: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('Weather API Response:', result);
        
        if (!result.success) {
            console.error('API Error:', result.error);
            throw new Error(result.error || 'Failed to fetch weather data');
        }
        
        const data = result.data;
        // Update weather display elements
        document.getElementById('weather-temp').textContent = `${data.temperature.toFixed(1)}Â°C`;
        document.getElementById('weather-humidity').textContent = `${data.humidity}%`;
        document.getElementById('weather-condition').textContent = data.condition;
        document.getElementById('weather-rainfall').textContent = `${data.rainfall} mm`;
        
        // Update location with city and country
        const locationText = data.location + (data.country ? `, ${data.country}` : '');
        document.getElementById('weather-city').textContent = locationText;
        
        // Show weather display and update sensor data
        document.getElementById('weather-display').classList.remove('hidden');
        sensorData.temperature = data.temperature;
        sensorData.humidity = data.humidity;
        updateSensorDisplay();
        calculateThreshold();
        showToast('Weather Updated', `Weather data loaded for ${location}`, 'success');
    } catch (e) {
        console.error('Weather Fetch Error:', e);
        showToast('Error', `Weather Error: ${e.message}`, 'error');
    } finally {
        btn.textContent = 'Get Weather';
        btn.disabled = false;
    }
}

// Sensor data functions
function updateSensorDisplay() {
    // Update temperature display
    document.getElementById('sensor-temp').textContent = 
        sensorData.temperature ? `${sensorData.temperature.toFixed(1)}Â°C` : '--Â°C';
    
    // Update humidity display
    document.getElementById('sensor-humidity').textContent = 
        sensorData.humidity ? `${sensorData.humidity.toFixed(1)}%` : '--%';
    
    // Update soil moisture display with Firebase percentage
    document.getElementById('sensor-soil').textContent = 
        sensorData.soilMoisturePercent !== undefined ? `${sensorData.soilMoisturePercent}%` : '--%';
    
    // Update water level display with Firebase percentage
    document.getElementById('sensor-water').textContent = 
        sensorData.waterLevelPercent !== undefined ? `${sensorData.waterLevelPercent}%` : '--%';
    const pumpIndicator = document.getElementById('pump-indicator');
    const pumpText = document.getElementById('pump-text');
    const pumpStatusDot = document.getElementById('pump-status-dot');
    const pumpStatusText = document.getElementById('pump-status-text');
    const pumpStatusDesc = document.getElementById('pump-status-desc');
    if (sensorData.pump) {
        pumpIndicator.className = 'status-dot active'; pumpText.textContent = 'ON';
        pumpStatusDot.className = 'status-dot active'; pumpStatusText.textContent = 'ON';
        pumpStatusDesc.textContent = 'Irrigation is active';
    } else {
        pumpIndicator.className = 'status-dot'; pumpText.textContent = 'OFF';
        pumpStatusDot.className = 'status-dot'; pumpStatusText.textContent = 'OFF';
        pumpStatusDesc.textContent = 'Irrigation is inactive';
    }
}

function updateConnectionStatus(status) {
    const statusElement = document.getElementById('connection-status');
    switch(status) {
        case 'connected': statusElement.textContent = 'Connected to Firebase'; statusElement.className = 'badge badge-success'; break;
        case 'connecting': statusElement.textContent = 'Connecting...'; statusElement.className = 'badge badge-warning'; break;
        case 'disconnected': statusElement.textContent = 'Add Firebase config to enable live IoT data'; statusElement.className = 'badge badge-error'; break;
    }
}

// Pump control
function toggleManualOverride(checked) {
    manualOverride = checked;
    const modeDisplay = document.getElementById('mode-display');
    const pumpOnBtn = document.getElementById('pump-on-btn');
    const pumpOffBtn = document.getElementById('pump-off-btn');
    
    // Update the display and button states
    modeDisplay.textContent = checked ? 'Manual' : 'Automatic';
    pumpOnBtn.disabled = !checked;
    pumpOffBtn.disabled = !checked;
    
    // Update Firebase actuators/manualOverride
    database.ref('actuators/manualOverride').set(checked)
        .then(() => {
            showToast(
                `Manual Override ${checked ? 'Enabled' : 'Disabled'}`,
                `Mode: ${checked ? 'Manual' : 'Auto'}`,
                'success'
            );
        })
        .catch(error => {
            console.error('Error updating manual override:', error);
            showToast('Error', 'Failed to update control mode', 'error');
            // Revert the switch if Firebase update fails
            document.getElementById('manual-override').checked = !checked;
            manualOverride = !checked;
            pumpOnBtn.disabled = !manualOverride;
            pumpOffBtn.disabled = !manualOverride;
            modeDisplay.textContent = manualOverride ? 'Manual' : 'Automatic';
        });
}

function setPump(state) {
    // Check if manual override is enabled
    if (!manualOverride) {
        showToast('Control Error', 'Enable manual override to control pump', 'error');
        return;
    }

    // Disable both buttons during operation
    const pumpOnBtn = document.getElementById('pump-on-btn');
    const pumpOffBtn = document.getElementById('pump-off-btn');
    pumpOnBtn.disabled = true;
    pumpOffBtn.disabled = true;

    // Show loading state
    showToast('Processing', `Setting pump ${state ? 'ON' : 'OFF'}...`, 'info');
    
    // Update Firebase actuators/pump
    database.ref('actuators/pump').set(state)
        .then(() => {
            // Update local state
            sensorData.pump = state;
            updateSensorDisplay();
            
            // Show success message
            showToast(
                `Success`,
                `Pump ${state ? 'started' : 'stopped'} successfully`,
                'success'
            );

            // Re-enable buttons
            pumpOnBtn.disabled = !manualOverride;
            pumpOffBtn.disabled = !manualOverride;
        })
        .catch(error => {
            console.error('Error updating pump status:', error);
            showToast('Control Error', `Failed to ${state ? 'start' : 'stop'} pump: ${error.message}`, 'error');
            
            // Re-enable buttons
            pumpOnBtn.disabled = !manualOverride;
            pumpOffBtn.disabled = !manualOverride;
            
            // Revert display state to match actual pump state
            updateSensorDisplay();
        });
}

// Calculations
function calculateThreshold() {
    const temperature = sensorData.temperature || 0;
    const humidity = sensorData.humidity || 0;
    const cropId = parseInt(document.getElementById('crop-select').value);
    const soilType = parseInt(document.getElementById('soil-select').value);
    const seedlingStage = parseInt(document.getElementById('stage-select').value);
    document.getElementById('temp-display').textContent = temperature || '--';
    document.getElementById('humidity-display').textContent = humidity || '--';
    if (temperature === 0 || humidity === 0) { document.getElementById('threshold-value').textContent = '--'; return; }
    const baseMoisture = cropMoistureRequirements[cropId].base;
    const tempFactor = 1 + (temperature - 25) * 0.02;
    const humidityFactor = 1 + (50 - humidity) * 0.01;
    const soilFactor = soilMoistureFactors[soilType];
    const stageFactor = stageMoistureMultipliers[seedlingStage];
    let threshold = baseMoisture * tempFactor * humidityFactor * soilFactor * stageFactor;
    threshold = Math.max(15, Math.min(70, threshold));
    threshold = Math.round(threshold * 10) / 10;
    cropData.threshold = threshold; document.getElementById('threshold-value').textContent = threshold;
    const score = 0.06 * (humidity - 60) - 0.035 * (temperature - 22);
    const prob = 1 / (1 + Math.exp(-score));
    const chance = Math.max(0, Math.min(100, Math.round(prob * 100)));
    const amount = chance >= 40 ? Math.round(((chance - 40) / 60) * 10 * 10) / 10 : 0;
    rainForecast = { chance, amount };
    document.getElementById('rain-chance').textContent = `${chance}%`;
    document.getElementById('rain-amount').textContent = `${amount} mm`;
    updateRecommendation();
}

function updateRecommendation() {
    const currentMoisture = parseFloat(document.getElementById('current-moisture').value) || 0;
    const threshold = cropData.threshold || 0;
    cropData.currentMoisture = currentMoisture;
    document.getElementById('current-moisture-display').textContent = currentMoisture ? `${currentMoisture}%` : '--%';
    document.getElementById('target-threshold-display').textContent = threshold ? `${threshold}%` : '--%';
    document.getElementById('rain-chance-display').textContent = `${rainForecast.chance}%`;
    document.getElementById('expected-rain-display').textContent = `${rainForecast.amount} mm`;
    if (threshold && currentMoisture) {
        const difference = currentMoisture - threshold;
        const comparisonEl = document.getElementById('moisture-comparison');
        const statusEl = document.getElementById('moisture-status');
        comparisonEl.classList.remove('hidden');
        if (Math.abs(difference) <= 2) { statusEl.textContent = 'âœ… Optimal Moisture Level'; statusEl.className = 'status-message success'; comparisonEl.className = 'moisture-comparison success'; }
        else if (difference > 2) { statusEl.textContent = `âš ï¸ Moisture Too High (+${difference.toFixed(1)}%)`; statusEl.className = 'status-message warning'; comparisonEl.className = 'moisture-comparison warning'; }
        else { statusEl.textContent = `ðŸš° Moisture Too Low (${difference.toFixed(1)}%)`; statusEl.className = 'status-message error'; comparisonEl.className = 'moisture-comparison error'; }
    } else { document.getElementById('moisture-comparison').classList.add('hidden'); }
    updateIrrigationRecommendation();
}

function updateIrrigationRecommendation() {
    const { currentMoisture, threshold } = cropData;
    const { chance, amount } = rainForecast;
    let recommendation;
    if (!threshold || !currentMoisture) {
        recommendation = { icon: 'alert-triangle', title: 'Enter Values Required', description: 'Complete the crop prediction form to see irrigation recommendations.', badge: 'PENDING', className: 'secondary' };
    } else {
        const difference = currentMoisture - threshold;
        const likelyRain = chance >= 60 && amount >= 0.5;
        if (Math.abs(difference) <= 2) { recommendation = { icon: 'check-circle', title: 'No Irrigation Needed', description: 'Soil moisture is optimal for current conditions and crop requirements.', badge: 'OPTIMAL', className: 'success' }; }
        else if (difference > 2) { recommendation = { icon: 'x-circle', title: 'Avoid Irrigation', description: 'Soil moisture is above optimal threshold. Overwatering can harm crops.', badge: 'AVOID', className: 'warning' }; }
        else {
            recommendation = likelyRain ? { icon: 'cloud-rain', title: 'Wait for Rain', description: `Rain expected within 2 hours (${chance}% chance, ${amount}mm). Irrigate only if it doesn't rain.`, badge: 'WAIT', className: 'warning' } : { icon: 'droplets', title: 'Irrigate Now', description: `Soil moisture is ${Math.abs(difference).toFixed(1)}% below threshold. Low rain probability (${chance}%).`, badge: 'IRRIGATE', className: 'error' };
        }
    }
    const displayEl = document.getElementById('recommendation-display');
    const iconEl = document.getElementById('recommendation-icon');
    const titleEl = document.getElementById('recommendation-title');
    const descEl = document.getElementById('recommendation-description');
    const badgeEl = document.getElementById('recommendation-badge');
    displayEl.className = `recommendation-display ${recommendation.className}`;
    iconEl.setAttribute('data-lucide', recommendation.icon);
    titleEl.textContent = recommendation.title;
    descEl.textContent = recommendation.description;
    badgeEl.textContent = recommendation.badge;
    badgeEl.className = `badge badge-${recommendation.className}`;
    if (window.lucide) lucide.createIcons();
}

// Toasts
function showToast(title, message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) {
        console.error('Toast container not found');
        return;
    }
    
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    // Create toast content
    const content = document.createElement('div');
    content.className = 'toast-content';
    
    const titleDiv = document.createElement('div');
    titleDiv.className = 'toast-title';
    titleDiv.textContent = title;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'toast-message';
    messageDiv.textContent = message;
    
    const closeButton = document.createElement('button');
    closeButton.className = 'toast-close';
    closeButton.textContent = 'Ã—';
    closeButton.onclick = () => toast.remove();
    
    // Assemble toast
    content.appendChild(titleDiv);
    content.appendChild(messageDiv);
    toast.appendChild(content);
    toast.appendChild(closeButton);
    
    // Add to container
    container.appendChild(toast);
    
    // Trigger animation
    requestAnimationFrame(() => {
        toast.classList.add('show');
    });
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

// Key handlers
document.addEventListener('DOMContentLoaded', () => {
    const loc = document.getElementById('location');
    if (loc) loc.addEventListener('keypress', e => { if (e.key === 'Enter') fetchWeather(); });
});

