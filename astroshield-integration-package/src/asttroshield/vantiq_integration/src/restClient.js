/**
 * REST client for Astroshield API integration with Vantiq
 */
const axios = require('axios');

class AstroshieldClient {
    constructor(baseUrl, apiKey) {
        this.baseUrl = baseUrl;
        this.apiKey = apiKey;
        this.axiosInstance = axios.create({
            baseURL: baseUrl,
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json'
            }
        });
    }
    
    /**
     * Get historical maneuvers for an object
     */
    async getObjectManeuverHistory(catalogId, startDate, endDate) {
        try {
            const response = await this.axiosInstance.get(`/api/v1/maneuvers`, {
                params: {
                    catalogId,
                    startDate,
                    endDate
                }
            });
            return response.data;
        } catch (error) {
            console.error('Error fetching maneuver history:', error);
            throw error;
        }
    }
    
    /**
     * Get object details
     */
    async getObjectDetails(catalogId) {
        try {
            const response = await this.axiosInstance.get(`/api/v1/objects/${catalogId}`);
            return response.data;
        } catch (error) {
            console.error('Error fetching object details:', error);
            throw error;
        }
    }
    
    /**
     * Get future observation windows
     */
    async getFutureObservationWindows(locationId, hours=24) {
        try {
            const response = await this.axiosInstance.get(`/api/v1/observations/forecast`, {
                params: {
                    locationId,
                    hours
                }
            });
            return response.data;
        } catch (error) {
            console.error('Error fetching observation windows:', error);
            throw error;
        }
    }
}

module.exports = AstroshieldClient; 