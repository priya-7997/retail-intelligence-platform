/**
 * API Communication Module for Retail Intelligence Platform
 * Handles all API requests and responses
 */

class APIClient {
    constructor() {
        this.baseURL = window.location.origin + '/api/v1';
        this.defaultHeaders = {
            'Content-Type': 'application/json'
        };
        this.requestTimeout = 30000; // 30 seconds
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            timeout: this.requestTimeout,
            headers: { ...this.defaultHeaders, ...options.headers },
            ...options
        };

        try {
            perfMonitor.start(`API_${endpoint}`);
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), config.timeout);
            
            const response = await fetch(url, {
                ...config,
                signal: controller.signal
            });

            clearTimeout(timeoutId);
            perfMonitor.end(`API_${endpoint}`);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return data;

        } catch (error) {
            perfMonitor.end(`API_${endpoint}`);
            
            if (error.name === 'AbortError') {
                throw new Error('Request timeout - please try again');
            }
            
            console.error(`API Error [${endpoint}]:`, error);
            throw error;
        }
    }

    // GET request
    async get(endpoint, params = {}) {
        const url = new URL(endpoint, this.baseURL);
        Object.keys(params).forEach(key => {
            if (params[key] !== null && params[key] !== undefined) {
                url.searchParams.append(key, params[key]);
            }
        });

        return this.request(url.pathname + url.search, { method: 'GET' });
    }

    // POST request
    async post(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    // PUT request
    async put(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }

    // DELETE request
    async delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }

    // File upload
    async uploadFile(endpoint, file, onProgress = null) {
        const formData = new FormData();
        formData.append('file', file);

        const config = {
            method: 'POST',
            body: formData,
            headers: {} // Don't set Content-Type for FormData
        };

        // Add progress tracking if callback provided
        if (onProgress && typeof onProgress === 'function') {
            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const percentComplete = (e.loaded / e.total) * 100;
                        onProgress(percentComplete);
                    }
                });

                xhr.addEventListener('load', () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        try {
                            const response = JSON.parse(xhr.responseText);
                            resolve(response);
                        } catch (e) {
                            reject(new Error('Invalid JSON response'));
                        }
                    } else {
                        reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`));
                    }
                });

                xhr.addEventListener('error', () => {
                    reject(new Error('Network error'));
                });

                xhr.addEventListener('timeout', () => {
                    reject(new Error('Request timeout'));
                });

                xhr.timeout = this.requestTimeout;
                xhr.open('POST', `${this.baseURL}${endpoint}`);
                xhr.send(formData);
            });
        }

        return this.request(endpoint, config);
    }
}

// Initialize API client
const api = new APIClient();

/**
 * Upload API functions
 */
const uploadAPI = {
    // Upload file for processing
    async uploadFile(file, onProgress) {
        return api.uploadFile('/upload', file, onProgress);
    },

    // Get upload status
    async getStatus(fileId) {
        return api.get(`/upload/status/${fileId}`);
    },

    // Get processing results
    async getResults(fileId) {
        return api.get(`/upload/results/${fileId}`);
    },

    // Delete uploaded file
    async deleteFile(fileId) {
        return api.delete(`/upload/${fileId}`);
    },

    // List uploaded files
    async listFiles() {
        return api.get('/upload/list');
    },

    // Validate CSV structure
    async validateCSV(file) {
        return api.uploadFile('/upload/validate', file);
    },

    // Get sample data format
    async getSampleFormat() {
        return api.get('/upload/sample');
    }
};

/**
 * Forecast API functions
 */
const forecastAPI = {
    // Generate auto forecast
    async autoForecast(fileId, periods = 30) {
        return api.post('/forecast/auto-forecast', { file_id: fileId, forecast_periods: periods });
    },

    // Generate forecast with specific model
    async generateForecast(request) {
        return api.post('/forecast/predict', request);
    },

    // Train specific model
    async trainModel(request) {
        return api.post('/forecast/train', request);
    },

    // Get training status
    async getTrainingStatus(jobId) {
        return api.get(`/forecast/train/status/${jobId}`);
    },

    // Get available models info
    async getModels() {
        return api.get('/forecast/models');
    },

    // Compare all models
    async compareModels(fileId) {
        return api.get(`/forecast/models/compare/${fileId}`);
    },

    // Delete trained model
    async deleteModel(modelId) {
        return api.delete(`/forecast/models/${modelId}`);
    }
};

/**
 * Insights API functions
 */
const insightsAPI = {
    // Generate comprehensive insights
    async generateInsights(request) {
        return api.post('/insights/generate', request);
    },

    // Get insights summary
    async getSummary(fileId) {
        return api.get(`/insights/summary/${fileId}`);
    },

    // Get insight types
    async getInsightTypes() {
        return api.get('/insights/types');
    },

    // Get specific recommendations
    async getRecommendations(fileId, category = 'all') {
        return api.post(`/insights/recommendations/${fileId}`, { category });
    },

    // Export insights report
    async exportReport(fileId, format = 'json') {
        return api.get(`/insights/export/${fileId}`, { format });
    }
};

/**
 * Dashboard API functions
 */
const dashboardAPI = {
    // Get dashboard overview
    async getOverview(fileId) {
        return api.get(`/dashboard/overview/${fileId}`);
    },

    // Get KPI metrics
    async getKPIs(fileId) {
        return api.get(`/dashboard/kpis/${fileId}`);
    },

    // Get chart data
    async getChartData(fileId, chartType = 'all') {
        return api.get(`/dashboard/charts/${fileId}`, { chart_type: chartType });
    },

    // Get active alerts
    async getAlerts(fileId) {
        return api.get(`/dashboard/alerts/${fileId}`);
    },

    // Get forecast preview
    async getForecastPreview(fileId) {
        return api.get(`/dashboard/forecast-preview/${fileId}`);
    }
};

/**
 * Utility functions for API handling
 */
const apiUtils = {
    // Polling function for long-running operations
    async pollStatus(getStatusFn, interval = 2000, maxAttempts = 150) {
        let attempts = 0;
        
        return new Promise((resolve, reject) => {
            const poll = async () => {
                try {
                    attempts++;
                    const result = await getStatusFn();
                    
                    if (result.status === 'completed') {
                        resolve(result);
                    } else if (result.status === 'failed') {
                        reject(new Error(result.error || 'Operation failed'));
                    } else if (attempts >= maxAttempts) {
                        reject(new Error('Operation timeout'));
                    } else {
                        setTimeout(poll, interval);
                    }
                } catch (error) {
                    reject(error);
                }
            };
            
            poll();
        });
    },

    // Retry function for failed requests
    async retry(fn, maxRetries = 3, delay = 1000) {
        let lastError;
        
        for (let i = 0; i <= maxRetries; i++) {
            try {
                return await fn();
            } catch (error) {
                lastError = error;
                
                if (i === maxRetries) {
                    throw lastError;
                }
                
                // Exponential backoff
                const waitTime = delay * Math.pow(2, i);
                await new Promise(resolve => setTimeout(resolve, waitTime));
            }
        }
    },

    // Batch request function
    async batchRequest(requests, concurrency = 3) {
        const results = [];
        const errors = [];
        
        for (let i = 0; i < requests.length; i += concurrency) {
            const batch = requests.slice(i, i + concurrency);
            
            const batchPromises = batch.map(async (request, index) => {
                try {
                    const result = await request();
                    results[i + index] = result;
                } catch (error) {
                    errors[i + index] = error;
                }
            });
            
            await Promise.all(batchPromises);
        }
        
        return { results, errors };
    },

    // Cache for API responses
    cache: new Map(),
    
    // Cached request function
    async cachedRequest(key, requestFn, ttl = 300000) { // 5 minutes default TTL
        const now = Date.now();
        const cached = this.cache.get(key);
        
        if (cached && (now - cached.timestamp) < ttl) {
            return cached.data;
        }
        
        try {
            const data = await requestFn();
            this.cache.set(key, { data, timestamp: now });
            return data;
        } catch (error) {
            // Return cached data if available, even if expired
            if (cached) {
                console.warn('Using expired cache due to request failure:', error);
                return cached.data;
            }
            throw error;
        }
    },

    // Clear cache
    clearCache(pattern = null) {
        if (pattern) {
            // Clear entries matching pattern
            for (const key of this.cache.keys()) {
                if (key.includes(pattern)) {
                    this.cache.delete(key);
                }
            }
        } else {
            // Clear all cache
            this.cache.clear();
        }
    }
};

/**
 * Error handling wrapper for API calls
 */
async function safeAPICall(apiFunction, errorMessage = 'Operation failed') {
    try {
        showLoading();
        const result = await apiFunction();
        hideLoading();
        return result;
    } catch (error) {
        hideLoading();
        console.error('API Error:', error);
        
        let userMessage = errorMessage;
        
        // Customize error message based on error type
        if (error.message.includes('timeout')) {
            userMessage = 'Request timed out. Please try again.';
        } else if (error.message.includes('Network')) {
            userMessage = 'Network error. Please check your connection.';
        } else if (error.message.includes('404')) {
            userMessage = 'Requested resource not found.';
        } else if (error.message.includes('500')) {
            userMessage = 'Server error. Please try again later.';
        } else if (error.message) {
            userMessage = error.message;
        }
        
        notifications.show(userMessage, 'error');
        throw error;
    }
}

/**
 * File upload with progress tracking
 */
async function uploadFileWithProgress(file, onProgress, onComplete, onError) {
    try {
        // Validate file before upload
        if (!file) {
            throw new Error('No file selected');
        }
        
        if (file.size > 50 * 1024 * 1024) { // 50MB limit
            throw new Error('File size must be less than 50MB');
        }
        
        const allowedTypes = ['.csv', '.xlsx', '.xls'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!allowedTypes.includes(fileExtension)) {
            throw new Error('File type not supported. Please upload CSV or Excel files.');
        }
        
        // Upload file
        const uploadResult = await uploadAPI.uploadFile(file, onProgress);
        
        if (!uploadResult.success) {
            throw new Error(uploadResult.error || 'Upload failed');
        }
        
        // Poll for processing status
        const statusResult = await apiUtils.pollStatus(
            () => uploadAPI.getStatus(uploadResult.file_id),
            2000, // Poll every 2 seconds
            150   // Max 5 minutes
        );
        
        if (onComplete) {
            onComplete(statusResult);
        }
        
        return statusResult;
        
    } catch (error) {
        if (onError) {
            onError(error);
        }
        throw error;
    }
}

/**
 * Generate forecast with automatic model selection
 */
async function generateAutoForecast(fileId, periods = 30) {
    try {
        notifications.show('Generating forecast...', 'info', 0, true);
        
        const result = await forecastAPI.autoForecast(fileId, periods);
        
        if (!result.success) {
            throw new Error(result.error || 'Forecast generation failed');
        }
        
        notifications.clear();
        notifications.show('Forecast generated successfully!', 'success');
        
        return result;
        
    } catch (error) {
        notifications.clear();
        handleError(error, 'generating forecast');
        throw error;
    }
}

/**
 * Generate comprehensive insights
 */
async function generateComprehensiveInsights(fileId, includeForecast = true) {
    try {
        notifications.show('Generating business insights...', 'info', 0, true);
        
        const request = {
            file_id: fileId,
            include_forecast: includeForecast
        };
        
        const result = await insightsAPI.generateInsights(request);
        
        if (!result.success) {
            throw new Error(result.error || 'Insights generation failed');
        }
        
        notifications.clear();
        notifications.show('Insights generated successfully!', 'success');
        
        return result;
        
    } catch (error) {
        notifications.clear();
        handleError(error, 'generating insights');
        throw error;
    }
}

// Export API modules for global use
window.api = api;
window.uploadAPI = uploadAPI;
window.forecastAPI = forecastAPI;
window.insightsAPI = insightsAPI;
window.dashboardAPI = dashboardAPI;
window.apiUtils = apiUtils;
window.safeAPICall = safeAPICall;
window.uploadFileWithProgress = uploadFileWithProgress;
window.generateAutoForecast = generateAutoForecast;
window.generateComprehensiveInsights = generateComprehensiveInsights; 
