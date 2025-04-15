/**
 * Extract features from Astroshield events for ML processing in Vantiq
 */

/**
 * Extract features from a maneuver detection event
 */
function extractManeuverFeatures(maneuverEvent) {
    // Basic features
    const features = {
        deltaV: maneuverEvent.deltaV,
        confidence: maneuverEvent.confidence,
        // Time-based features
        hourOfDay: new Date(maneuverEvent.detectionTime).getHours(),
        dayOfWeek: new Date(maneuverEvent.detectionTime).getDay(),
        // Categorical encoding
        isStationkeeping: maneuverEvent.maneuverType === "STATIONKEEPING" ? 1 : 0,
        isOrbitMaintenance: maneuverEvent.maneuverType === "ORBIT_MAINTENANCE" ? 1 : 0,
        isOrbitAdjustment: maneuverEvent.maneuverType === "ORBIT_ADJUSTMENT" ? 1 : 0,
        isMajorManeuver: maneuverEvent.maneuverType === "MAJOR_MANEUVER" ? 1 : 0
    };
    
    return features;
}

/**
 * Extract features from an observation window event
 */
function extractObservationFeatures(observationEvent) {
    // Basic features
    const features = {
        qualityScore: observationEvent.qualityScore,
        // Location features
        latitude: observationEvent.location.latitude,
        longitude: observationEvent.location.longitude,
        // Time features
        durationMinutes: observationEvent.observationWindow.durationMinutes,
        startHourUTC: new Date(observationEvent.observationWindow.startTime).getUTCHours(),
        // Target features
        altitudeKm: observationEvent.targetObject.altitudeKm,
        // Categorical encoding
        isExcellent: observationEvent.qualityCategory === "EXCELLENT" ? 1 : 0,
        isGood: observationEvent.qualityCategory === "GOOD" ? 1 : 0,
        isFair: observationEvent.qualityCategory === "FAIR" ? 1 : 0,
        isPoor: observationEvent.qualityCategory === "POOR" ? 1 : 0
    };
    
    return features;
}

module.exports = {
    extractManeuverFeatures,
    extractObservationFeatures
}; 