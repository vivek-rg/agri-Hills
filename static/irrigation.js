// Function to calculate moisture threshold
function calculateThreshold() {
    const cropSelect = document.getElementById('crop-select');
    const soilSelect = document.getElementById('soil-select');
    const stageSelect = document.getElementById('stage-select');
    const currentMoistureInput = document.getElementById('current-moisture');
    
    if (!cropSelect || !soilSelect || !stageSelect || !currentMoistureInput) {
        console.error('Required form elements not found');
        return;
    }

    const cropId = cropSelect.value;
    const soilType = soilSelect.value;
    const seedlingStage = stageSelect.value;
    const currentMoisture = parseFloat(currentMoistureInput.value) || 0;

    // Get base moisture requirement for the crop
    const baseMoisture = cropMoistureRequirements[cropId]?.base || 35;
    
    // Apply soil type and growth stage factors
    const soilFactor = soilMoistureFactors[soilType] || 1.0;
    const stageFactor = stageMoistureMultipliers[seedlingStage] || 1.0;
    
    // Get current temperature and humidity
    const temperature = sensorData.temperature || 25;
    const humidity = sensorData.humidity || 60;
    
    // Calculate final threshold
    const threshold = (baseMoisture * soilFactor * stageFactor * 
        (1 + (temperature - 25) * 0.01) * 
        (1 + (humidity - 60) * 0.005)).toFixed(1);

    // Update threshold display
    document.getElementById('threshold-value').textContent = threshold;
    document.getElementById('temp-display').textContent = temperature.toFixed(1);
    document.getElementById('humidity-display').textContent = humidity.toFixed(1);
    
    // Calculate and display recommendation
    const rainChance = parseFloat(document.getElementById('rain-chance').textContent) || 0;
    
    // Update recommendation if we have current moisture value
    if (currentMoisture > 0) {
        updateRecommendationDisplay(currentMoisture, parseFloat(threshold), rainChance);
    }

    return parseFloat(threshold);
}

// Event listeners for form inputs
document.addEventListener('DOMContentLoaded', function() {
    const inputs = [
        'crop-select',
        'soil-select',
        'stage-select',
        'current-moisture'
    ];
    
    inputs.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('change', function() {
                calculateThreshold();
            });
            if (id === 'current-moisture') {
                element.addEventListener('input', function() {
                    calculateThreshold();
                });
            }
        }
    });
    
    // Initial calculation
    calculateThreshold();
});