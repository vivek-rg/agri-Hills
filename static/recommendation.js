// Function to update recommendation display
function updateRecommendationDisplay(currentMoisture, threshold, rainChance) {
    const recommendationDiv = document.getElementById('recommendation-display');
    const iconElement = document.getElementById('recommendation-icon');
    const titleElement = document.getElementById('recommendation-title');
    const descElement = document.getElementById('recommendation-description');
    const badgeElement = document.getElementById('recommendation-badge');

    // Remove pending state
    recommendationDiv.classList.remove('pending');
    
    // Calculate the moisture difference
    const difference = currentMoisture - threshold;
    const likelyRain = rainChance >= 60;
    
    let recommendation = {
        icon: '',
        title: '',
        description: '',
        badge: '',
        className: ''
    };

    if (Math.abs(difference) <= 2) {
        recommendation = {
            icon: 'check-circle',
            title: 'No Irrigation Needed',
            description: 'Soil moisture is optimal for current conditions and crop requirements.',
            badge: 'OPTIMAL',
            className: 'success'
        };
    } else if (difference > 2) {
        recommendation = {
            icon: 'x-circle',
            title: 'Avoid Irrigation',
            description: 'Soil moisture is above optimal threshold. Overwatering can harm crops.',
            badge: 'AVOID',
            className: 'warning'
        };
    } else {
        if (likelyRain) {
            recommendation = {
                icon: 'cloud-rain',
                title: 'Wait for Rain',
                description: `Rain expected within 2 hours (${rainChance}% chance). Irrigate only if it doesn't rain.`,
                badge: 'WAIT',
                className: 'warning'
            };
        } else {
            recommendation = {
                icon: 'droplets',
                title: 'Irrigate Now',
                description: `Soil moisture is ${Math.abs(difference).toFixed(1)}% below threshold. Low rain probability (${rainChance}%).`,
                badge: 'IRRIGATE',
                className: 'error'
            };
        }
    }

    // Update display
    recommendationDiv.className = `recommendation-display ${recommendation.className}`;
    iconElement.setAttribute('data-lucide', recommendation.icon);
    titleElement.textContent = recommendation.title;
    descElement.textContent = recommendation.description;
    badgeElement.textContent = recommendation.badge;
    badgeElement.className = `badge badge-${recommendation.className}`;
    
    // Update icons
    if (window.lucide) lucide.createIcons();
}