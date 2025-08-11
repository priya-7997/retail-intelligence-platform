// Global fetch wrapper for robust error handling
async function robustFetch(url, options = {}) {
    try {
        const response = await fetch(url, options);
        if (!response.ok) {
            let errorMsg = `API error: ${response.status}`;
            let errorDetail = '';
            try {
                const data = await response.clone().json();
                errorDetail = data.detail || data.error || data.message || '';
                errorMsg = errorDetail ? `${errorMsg}: ${errorDetail}` : errorMsg;
                console.error('[API ERROR]', url, data);
            } catch (jsonErr) {
                // Not JSON or no detail
            }
            notifications.show(errorMsg, 'error');
            throw new Error(errorMsg);
        }
        return response;
    } catch (error) {
        if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
            notifications.show('Connection to backend lost. Please check your server.', 'error');
        } else {
            notifications.show(error.message, 'error');
        }
        throw error;
    }
}

// forecastAPI is provided globally by api.js; do not redeclare here

class RetailIntelligenceApp {
    constructor() {
        this.currentSection = 'dashboard';
        this.currentFileId = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadFileList();
        this.showSection('dashboard');
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const section = e.currentTarget.dataset.section;
                console.debug('[DEBUG] Navigation clicked:', section);
                this.showSection(section);
            });
        });

        // File upload
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');

        if (fileInput && uploadArea) {
            uploadArea.addEventListener('click', () => {
                console.debug('[DEBUG] Upload area clicked');
                fileInput.click();
            });
            fileInput.addEventListener('change', (e) => {
                console.debug('[DEBUG] File input changed', e.target.files);
                this.handleFileUpload(e);
            });
            // Drag and drop
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                console.debug('[DEBUG] File(s) dropped', files);
                if (files.length > 0) {
                    this.uploadFile(files[0]);
                }
            });
        }

        // Generate buttons
        const generateInsightsBtn = document.getElementById('generateInsights');
        const generateForecastBtn = document.getElementById('generateForecast');
        const trainModelBtn = document.getElementById('trainModelBtn');

        if (generateInsightsBtn) {
            generateInsightsBtn.addEventListener('click', () => {
                console.debug('[DEBUG] Generate Insights button clicked');
                this.generateInsights();
            });
        }
        if (generateForecastBtn) {
            generateForecastBtn.addEventListener('click', () => {
                console.debug('[DEBUG] Generate Forecast button clicked');
                this.generateForecast();
            });
        }
        if (trainModelBtn) {
            trainModelBtn.addEventListener('click', () => {
                console.debug('[DEBUG] Train Model button clicked');
                this.trainSelectedModel();
            });
        }
    }

    async trainSelectedModel() {
        const fileSelector = document.getElementById('forecastFileSelector');
        const modelSelector = document.getElementById('modelType');
        const fileId = fileSelector?.value;
        const modelType = modelSelector?.value;

        if (!fileId) {
            notifications.show('Please select a processed file to train.', 'warning');
            return;
        }
        if (!modelType || modelType === 'auto') {
            notifications.show('Please select a specific model to train.', 'warning');
            return;
        }

        try {
            show(document.getElementById('forecast-loading'));
            hide(document.getElementById('forecast-results'));
            notifications.show('Training model. This may take a minute...', 'info');
            const result = await forecastAPI.trainModel({
                file_id: fileId,
                model_type: modelType
            });
            if (result.success) {
                notifications.show('Model training started successfully!', 'success');
            } else {
                notifications.show(result.message || 'Model training failed', 'error');
            }
        } catch (error) {
            console.error('Model training error:', error);
            notifications.show(`Model training failed: ${error.message}`, 'error');
        } finally {
            hide(document.getElementById('forecast-loading'));
        }
    }

    showSection(sectionName) {
        // Update navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active')
        });
        document.querySelector(`[data-section="${sectionName}"]`).classList.add('active');

        // Update content
        document.querySelectorAll('.content-section').forEach(section => {
            section.classList.remove('active');
        });
        document.getElementById(`${sectionName}-section`).classList.add('active');

        this.currentSection = sectionName;

        // Load section-specific data
        this.loadSectionData(sectionName);
    }

    async loadSectionData(sectionName) {
        switch (sectionName) {
            case 'dashboard':
                await this.loadDashboard();
                break;
            case 'insights':
                this.loadInsightsSection();
                break;
            case 'forecast':
                this.loadForecastSection();
                break;
        }
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (file) {
            await this.uploadFile(file);
        }
    }

    async uploadFile(file) {
        try {
            this.showUploadProgress();
            
            const result = await uploadFileWithProgress(
                file,
                (progress) => this.updateUploadProgress(progress),
                (result) => this.handleUploadComplete(result),
                (error) => this.handleUploadError(error)
            );

        } catch (error) {
            this.handleUploadError(error);
        }
    }

    showUploadProgress() {
        hide(document.getElementById('uploadArea'));
        show(document.getElementById('uploadProgress'));
    }

    updateUploadProgress(percentage) {
        const progressFill = document.getElementById('progressFill');
        const progressPercentage = document.querySelector('.progress-percentage');
        
        if (progressFill) {
            progressFill.style.width = `${percentage}%`;
        }
        
        if (progressPercentage) {
            progressPercentage.textContent = `${Math.round(percentage)}%`;
        }

         // Update progress steps
        const steps = document.querySelectorAll('.progress-step');
        steps.forEach((step, index) => {
            if (percentage > index * 25) {
                step.classList.add('active');
            }
        });
    }

    handleUploadComplete(result) {
        hide(document.getElementById('uploadProgress'));
        show(document.getElementById('uploadResults'));
        
        // Display results
        this.displayUploadResults(result);
        
        // Refresh file list
        this.loadFileList();
        
        notifications.show('File processed successfully!', 'success');
    }

    handleUploadError(error) {
        hide(document.getElementById('uploadProgress'));
        show(document.getElementById('uploadArea'));
        
        notifications.show(`Upload failed: ${error.message}`, 'error');
    }

    displayUploadResults(result) {
        const resultsContainer = document.getElementById('uploadResults');
        if (!resultsContainer) return;

        const results = result.results || {};
        
        resultsContainer.innerHTML = `
            <div class="results-header">
                <div class="results-icon">
                    <i class="fas fa-check-circle"></i>
                </div>
                <div class="results-info">
                    <h3>File Processed Successfully</h3>
                    <p>Your data has been analyzed and is ready for insights and forecasting.</p>
                </div>
            </div>
            
            <div class="results-grid">
                <div class="result-card">
                    <h4>Data Overview</h4>
                    <div class="result-item">
                        <span class="result-label">Total Records</span>
                        <span class="result-value">${formatNumber(results.processed_shape?.[0] || 0)}</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Columns</span>
                        <span class="result-value">${results.processed_shape?.[1] || 0}</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Data Quality</span>
                        <span class="result-value">${Math.round(results.quality_score || 0)}%</span>
                    </div>
                </div>
                
                <div class="result-card">
                    <h4>Detected Columns</h4>
                    <div class="result-item">
                        <span class="result-label">Date Column</span>
                        <span class="result-value">${results.column_analysis?.date_column || 'Not detected'}</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Sales Column</span>
                        <span class="result-value">${results.column_analysis?.sales_column || 'Not detected'}</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Product Column</span>
                        <span class="result-value">${results.column_analysis?.product_column || 'Not detected'}</span>
                    </div>
                </div>
            </div>
            
            <div class="recommendations-list">
                <h4>Recommendations</h4>
                ${(results.recommendations || []).map(rec => `
                    <div class="recommendation-item">
                        <i class="fas fa-lightbulb recommendation-icon"></i>
                        <span class="recommendation-text">${rec}</span>
                    </div>
                `).join('')}
            </div>
            
            <div class="result-actions">
                <button class="btn-primary" onclick="app.showSection('insights')">
                    <i class="fas fa-lightbulb"></i>
                    Generate Insights
                </button>
                <button class="btn-primary" onclick="app.showSection('forecast')">
                    <i class="fas fa-crystal-ball"></i>
                    Create Forecast
                </button>
            </div>
        `;
    }

    async loadFileList() {
        try {
            // Use the correct endpoint for listing files
            const response = await robustFetch('/api/v1/upload/upload/list');
            const result = await response.json();
            if (result.success) {
                this.updateFileSelectors(result.files);
            }
        } catch (error) {
            console.error('Failed to load file list:', error);
        }
    }

    updateFileSelectors(files) {
        const selectors = [
            'fileSelector',
            'insightsFileSelector', 
            'forecastFileSelector'
        ];

        selectors.forEach(selectorId => {
            const selector = document.getElementById(selectorId);
            if (selector) {
                selector.innerHTML = '<option value="">Select processed file...</option>';
                
                files.forEach(file => {
                    if (file.status === 'completed') {
                        const option = document.createElement('option');
                        option.value = file.file_id;
                        option.textContent = `${file.filename} (${formatRelativeTime(file.completed_at)})`;
                        selector.appendChild(option);
                    }
                });

                // Auto-select the most recent file
                if (files.length > 0 && !this.currentFileId) {
                    const latestFile = files
                        .filter(f => f.status === 'completed')
                        .sort((a, b) => new Date(b.completed_at) - new Date(a.completed_at))[0];
                    
                    if (latestFile) {
                        selector.value = latestFile.file_id;
                        this.currentFileId = latestFile.file_id;
                    }
                }
            }
        });
    }

    async loadDashboard() {
        const fileSelector = document.getElementById('fileSelector');
        const fileId = fileSelector?.value || this.currentFileId;
        if (!fileId) {
            this.showWelcomeState();
            return;
        }
        try {
            showLoading('Loading Dashboard...', 'Fetching your business insights');
            // Use correct backend endpoint
            const response = await robustFetch(`/api/v1/dashboard/overview/${encodeURIComponent(fileId)}`);
            const result = await response.json();
            if (result.success) {
                this.displayDashboard(result);
                hide(document.getElementById('welcome-state'));
            }
            
        } catch (error) {
            console.error('Failed to load dashboard:', error);
        } finally {
            hideLoading();
        }
    }

    showWelcomeState() {
        show(document.getElementById('welcome-state'));
        hide(document.getElementById('kpi-section'));
        hide(document.getElementById('charts-section'));
        hide(document.getElementById('alerts-section'));
        hide(document.getElementById('insights-preview'));
    }

    displayDashboard(dashboardData) {
        this.displayKPIs(dashboardData.kpis);
        this.displayCharts(dashboardData.charts);
        this.displayAlerts(dashboardData.alerts);
        this.displayInsightsPreview(dashboardData.recent_insights);
    }

    displayKPIs(kpis) {
        const kpiSection = document.getElementById('kpi-section');
        if (!kpiSection || !kpis) return;

        kpiSection.innerHTML = kpis.map(kpi => `
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">${kpi.title}</span>
                    <span class="kpi-icon">${kpi.icon}</span>
                </div>
                <div class="kpi-value">${kpi.value}</div>
                <div class="kpi-description">${kpi.description}</div>
                ${kpi.trend ? `
                    <div class="kpi-trend ${kpi.trend.trend}">
                        <i class="fas fa-arrow-${kpi.trend.trend === 'up' ? 'up' : kpi.trend.trend === 'down' ? 'down' : 'right'}"></i>
                        ${formatPercentage(Math.abs(kpi.trend.change))}
                    </div>
                ` : ''}
            </div>
        `).join('');

        show(kpiSection);
    }

    displayCharts(charts) {
        const chartsSection = document.getElementById('charts-section');
        if (!chartsSection || !charts) return;

        // Clear existing charts
        chartManager.destroyAllCharts();

        // Display sales trend
        if (charts.sales_trend) {
            this.createSalesTrendChart(charts.sales_trend);
        }

        // Display other charts
        if (charts.product_performance) {
            this.createProductChart(charts.product_performance);
        }

        if (charts.weekly_pattern) {
            this.createWeeklyChart(charts.weekly_pattern);
        }

        if (charts.revenue_distribution) {
            this.createDistributionChart(charts.revenue_distribution);
        }

        show(chartsSection);
    }

    createSalesTrendChart(chartData) {
        if (!chartData.data || !Array.isArray(chartData.data)) return;

        const data = {
            labels: chartData.data.map(d => formatDate(d.date)),
            datasets: [{
                label: 'Daily Sales',
                data: chartData.data.map(d => d.sales),
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.4,
                fill: true
            }]
        };

        chartManager.createLineChart('salesTrendChart', data, {
            title: 'Sales Trend Analysis'
        });
    }

    createProductChart(chartData) {
        if (!chartData.data || !Array.isArray(chartData.data)) return;

        const data = {
            labels: chartData.data.map(d => d.product),
            datasets: [{
                label: 'Revenue',
                data: chartData.data.map(d => d.total_sales),
                backgroundColor: chartManager.colors
            }]
        };

        chartManager.createBarChart('productChart', data, {
            title: 'Top Products by Revenue'
        });
    }

    createWeeklyChart(chartData) {
        if (!chartData.data || !Array.isArray(chartData.data)) return;

        const data = {
            labels: chartData.data.map(d => d.day),
            datasets: [{
                label: 'Average Sales',
                data: chartData.data.map(d => d.avg_sales),
                backgroundColor: '#8b5cf6'
            }]
        };

        chartManager.createBarChart('weeklyChart', data, {
            title: 'Weekly Sales Pattern'
        });
    }

    createDistributionChart(chartData) {
        if (!chartData.data || !Array.isArray(chartData.data)) return;

        const data = {
            labels: chartData.data.map(d => d.range),
            datasets: [{
                label: 'Transactions',
                data: chartData.data.map(d => d.count),
                backgroundColor: '#f59e0b'
            }]
        };

        chartManager.createBarChart('distributionChart', data, {
            title: 'Transaction Value Distribution'
        });
    }

    displayAlerts(alerts) {
        const alertsSection = document.getElementById('alerts-section');
        const alertsList = document.getElementById('alerts-list');
        
        if (!alertsList || !alerts || alerts.length === 0) {
            hide(alertsSection);
            return;
        }

        alertsList.innerHTML = alerts.map(alert => `
            <div class="alert ${alert.type}">
                <div class="alert-header">
                    <i class="fas fa-${this.getAlertIcon(alert.type)}"></i>
                    ${alert.title}
                </div>
                <div class="alert-message">${alert.message}</div>
                ${alert.action ? `<div class="alert-action">Action: ${alert.action}</div>` : ''}
            </div>
        `).join('');

        show(alertsSection);
    }

    getAlertIcon(type) {
        const icons = {
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle',
            success: 'check-circle'
        };
        return icons[type] || 'info-circle';
    }

    displayInsightsPreview(insights) {
        const insightsPreview = document.getElementById('insights-preview');
        const insightsCards = document.getElementById('insights-cards');
        if (!insightsCards || !insights || insights.length === 0) {
            hide(insightsPreview);
            return;
        }
        insightsCards.innerHTML = insights.map(insight => `
            <div class="insight-preview-card ${insight.type}">
                <div class="insight-preview-header">
                    <i class="fas fa-${this.getInsightIcon(insight.type)}"></i>
                    <span class="insight-preview-title">${insight.title}</span>
                </div>
                <div class="insight-preview-content">
                    <b>What this means for you:</b> ${insight.action || 'Take action based on this insight to improve your business.'}<br>
                    <span>${insight.content}</span>
                </div>
                <span class="insight-priority ${insight.priority}">${insight.priority} priority</span>
            </div>
        `).join('');
        show(insightsPreview);
    }

    getInsightIcon(type) {
        const icons = {
            executive: 'chart-line',
            trend: 'trending-up',
            opportunity: 'lightbulb'
        };
        return icons[type] || 'info-circle';
    }

    async generateInsights() {
        const fileSelector = document.getElementById('insightsFileSelector');
        const fileId = fileSelector?.value;
        
        if (!fileId) {
            notifications.show('Please select a processed file', 'warning');
            return;
        }

        try {
            show(document.getElementById('insights-loading'));
            hide(document.getElementById('insights-dashboard'));
            
            const result = await generateComprehensiveInsights(fileId);
            
            if (result.success) {
                this.displayInsightsDashboard(result.insights);
            }
            
        } catch (error) {
            console.error('Insights generation error:', error);
        } finally {
            hide(document.getElementById('insights-loading'));
        }
    }

    displayInsightsDashboard(insights) {
        // Display executive summary
        this.displayExecutiveSummary(insights.executive_summary);
        // Display action items
        this.displayActionItems(insights.action_items);
        // Display insights by category
        this.displayInsightsByCategory(insights);
        // Display risk alerts if any
        if (insights.risk_alerts && insights.risk_alerts.length > 0) {
            this.displayRiskAlerts(insights.risk_alerts);
        }
        // Add more customer-friendly, actionable insights
        const extraAnalysis = document.getElementById('extraAnalysis');
        if (extraAnalysis) {
            extraAnalysis.innerHTML = `
                <h3>Additional Analysis</h3>
                <ul>
                    <li><b>Seasonality Detection:</b> Identify seasonal patterns in your sales data for better planning.</li>
                    <li><b>Top Products:</b> See which products are driving your revenue and focus marketing efforts.</li>
                    <li><b>Customer Segmentation:</b> Discover key customer groups and tailor promotions.</li>
                    <li><b>Sales Anomalies:</b> Get alerts for unusual sales spikes or drops.</li>
                    <li><b>Promotion Impact:</b> Analyze how discounts and campaigns affect your sales.</li>
                </ul>
                <p>Contact our analytics team for custom reports and deeper insights!</p>
            `;
        }
        show(document.getElementById('insights-dashboard'));
    }

    displayExecutiveSummary(summary) {
        const container = document.getElementById('executiveSummary');
        if (!container || !summary) return;

        container.innerHTML = summary.map(item => `
            <div class="insight-item">
                <i class="fas fa-chart-line"></i>
                <span><b>${item.title}</b>: ${item.detail}</span>
                <div class="insight-context">${item.context || ''}</div>
                <div class="insight-action"><i class="fas fa-lightbulb"></i> ${item.action || ''}</div>
            </div>
        `).join('');
    }

    displayActionItems(actions) {
        const container = document.getElementById('actionItems');
        if (!container || !actions) return;

        container.innerHTML = actions.map(action => `
            <div class="insight-item">
                <i class="fas fa-tasks"></i>
                <span>${action}</span>
            </div>
        `).join('');
    }

    displayInsightsByCategory(insights) {
        const categories = [
            { id: 'salesInsights', key: 'sales_insights', icon: 'trending-up', label: 'Sales Insights' },
            { id: 'seasonalInsights', key: 'seasonality_insights', icon: 'calendar-alt', label: 'Seasonality' },
            { id: 'inventoryInsights', key: 'inventory_alerts', icon: 'boxes', label: 'Inventory Alerts' },
            { id: 'revenueInsights', key: 'revenue_opportunities', icon: 'dollar-sign', label: 'Revenue Opportunities' }
        ];

        categories.forEach(category => {
            const container = document.getElementById(category.id);
            const data = insights[category.key];
            if (container && data) {
                container.innerHTML = data.map(item => `
                    <div class="insight-item" style="margin-bottom: 1em; padding: 1em; border-radius: 8px; background: #f9f9f9; box-shadow: 0 1px 4px rgba(0,0,0,0.04);">
                        <div style="display: flex; align-items: center;">
                            <i class="fas fa-${category.icon}" style="font-size: 1.5em; color: #10b981; margin-right: 0.5em;"></i>
                            <span style="font-weight: bold; font-size: 1.1em;">${item.title || item}</span>
                        </div>
                        <div class="insight-context" style="margin: 0.5em 0; color: #555;">${item.context || ''}</div>
                        <div class="insight-action" style="color: #f59e0b;"><i class="fas fa-lightbulb"></i> ${item.action || ''}</div>
                    </div>
                `).join('');
            }
        });
    }

    displayRiskAlerts(alerts) {
        const container = document.getElementById('riskAlerts');
        const card = document.getElementById('riskAlertsCard');
        
        if (!container || !alerts) return;

        container.innerHTML = alerts.map(alert => `
            <div class="risk-alert-item">
                <div class="risk-alert-icon">
                    <i class="fas fa-exclamation-triangle"></i>
                </div>
                <div class="risk-alert-content">
                    <div class="risk-alert-title">Risk Alert</div>
                    <div class="risk-alert-message">${alert}</div>
                    <div class="risk-alert-action">Review immediately</div>
                </div>
            </div>
        `).join('');

        show(card);
    }

    async generateForecast() {
        const fileSelector = document.getElementById('forecastFileSelector');
        const periodsSelector = document.getElementById('forecastDays');
        const modelSelector = document.getElementById('modelType');
        let frequencySelector = document.getElementById('forecastFrequency');

        // If frequency dropdown doesn't exist, create it dynamically (for patching)
        if (!frequencySelector) {
            frequencySelector = document.createElement('select');
            frequencySelector.id = 'forecastFrequency';
            frequencySelector.innerHTML = `
                <option value="daily" selected>Daily</option>
                <option value="weekly">Weekly</option>
                <option value="monthly">Monthly</option>
            `;
            // Insert after modelSelector
            if (modelSelector && modelSelector.parentNode) {
                modelSelector.parentNode.parentNode.insertBefore(frequencySelector, modelSelector.parentNode.nextSibling);
            }
        }

        const fileId = fileSelector?.value;
        const periods = parseInt(periodsSelector?.value || '30');
        const modelType = modelSelector?.value || 'auto';
        const frequency = frequencySelector?.value || 'daily';

        // Validate required fields
        if (!fileId) {
            notifications.show('Please select a processed file', 'warning');
            return;
        }
        if (!periods || isNaN(periods) || periods < 1) {
            notifications.show('Please enter a valid forecast period (days)', 'warning');
            return;
        }
        if (modelType !== 'auto' && !modelType) {
            notifications.show('Please select a forecast model', 'warning');
            return;
        }
        if (!frequency) {
            notifications.show('Please select a forecast frequency', 'warning');
            return;
        }

        try {
            show(document.getElementById('forecast-loading'));
            hide(document.getElementById('forecast-results'));
            let requestPayload = {
                file_id: fileId,
                forecast_periods: periods,
                frequency: frequency
            };
            if (modelType !== 'auto') {
                requestPayload.model_type = modelType;
            }
            console.log('Sending forecast request:', JSON.stringify(requestPayload, null, 2));
            const response = await robustFetch('/api/v1/forecast/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestPayload)
            });
            const result = await response.json();
            if (result.success) {
                this.displayForecastResults(result);
            } else {
                notifications.show(result.message || result.detail || 'Forecast generation failed', 'error');
                console.error('Forecast API error:', result);
            }
        } catch (error) {
            console.error('Forecast generation error:', error);
            notifications.show(`Forecast generation failed: ${error.message}`, 'error');
        } finally {
            hide(document.getElementById('forecast-loading'));
        }
    }

    displayForecastResults(result) {
        // Display forecast summary
        this.displayForecastSummary(result);
        
        // Display model information
        this.displayModelInfo(result);
        
        // Display forecast chart
        this.displayForecastChart(result.forecast);
        
        // Display forecast table
        this.displayForecastTable(result.forecast);

        show(document.getElementById('forecast-results'));
    }

    displayForecastSummary(result) {
        const container = document.getElementById('forecastSummaryContent');
        if (!container) return;

        const summary = result.forecast_summary || {};
        container.innerHTML = `
            <div class="summary-metric">
                <span class="metric-label">Total Forecasted Revenue</span>
                <span class="metric-value">${summary.total_forecasted_revenue || 'N/A'}</span>
            </div>
            <div class="summary-metric">
                <span class="metric-label">Average Daily Revenue</span>
                <span class="metric-value">${summary.average_daily_revenue || 'N/A'}</span>
            </div>
            <div class="summary-metric">
                <span class="metric-label">Trend Direction</span>
                <span class="metric-value">${capitalizeFirst(summary.trend_direction || 'stable')}</span>
            </div>
            <div class="summary-metric">
                <span class="metric-label">Forecast Period</span>
                <span class="metric-value">${summary.forecast_period_days || 30} days</span>
            </div>
            <div class="summary-metric">
                <span class="metric-label">Key Takeaway</span>
                <span class="metric-value">${summary.key_takeaway || 'Review the forecast to plan your inventory and promotions.'}</span>
            </div>
        `;
    }

    displayModelInfo(result) {
        const container = document.getElementById('modelInfoContent');
        if (!container) return;

        container.innerHTML = `
            <div class="model-badge">
                <i class="fas fa-brain"></i>
                ${result.model_used || 'Unknown'}
            </div>
            <div class="confidence-meter">
                <span class="confidence-label">Confidence: ${formatPercentage(result.selection_confidence * 100 || 80)}</span>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${(result.selection_confidence || 0.8) * 100}%"></div>
                </div>
            </div>
            ${result.training_metrics ? `
                <div class="metric-item">
                    <span>Accuracy (MAE):</span>
                    <span>${result.training_metrics.mae?.toFixed(2) || 'N/A'}</span>
                </div>
            ` : ''}
        `;
    }

    displayForecastChart(forecastData) {
        if (!forecastData || !Array.isArray(forecastData)) return;

        const data = {
            labels: forecastData.map(d => formatDate(d.ds)),
            datasets: [{
                label: 'Forecast',
                data: forecastData.map(d => d.yhat),
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                tension: 0.4,
                fill: true
            }, {
                label: 'Upper Bound',
                data: forecastData.map(d => d.yhat_upper),
                borderColor: '#f59e0b',
                backgroundColor: 'transparent',
                borderDash: [5, 5],
                pointRadius: 0
            }, {
                label: 'Lower Bound',
                data: forecastData.map(d => d.yhat_lower),
                borderColor: '#f59e0b',
                backgroundColor: 'transparent',
                borderDash: [5, 5],
                pointRadius: 0
            }]
        };

            const chartElem = document.getElementById('forecastChart');
            // If a previous chart exists, destroy it and reset the canvas
            if (this.forecastChart) {
                this.forecastChart.destroy();
                // Remove and recreate the canvas to fully reset Chart.js state
                const parent = chartElem.parentNode;
                parent.removeChild(chartElem);
                const newCanvas = document.createElement('canvas');
                newCanvas.id = 'forecastChart';
                parent.appendChild(newCanvas);
            }
            const ctx = document.getElementById('forecastChart').getContext('2d');
            this.forecastChart = new Chart(ctx, {
                type: 'line',
                data: data,
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: `Forecast`
                        },
                        legend: {
                            display: true
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Date'
                            }
                        },
                        y: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Value'
                            }
                        }
                    }
                }
            });
    }

    displayForecastTable(forecastData) {
        const table = document.getElementById('forecastTable');
        if (!table || !forecastData) return;

        const tableHTML = `
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Predicted Sales</th>
                    <th>Lower Bound</th>
                    <th>Upper Bound</th>
                </tr>
            </thead>
            <tbody>
                ${forecastData.map(row => `
                    <tr>
                        <td class="date-cell">${formatDate(row.ds)}</td>
                        <td class="value-cell">${formatCurrency(row.yhat)}</td>
                        <td class="confidence-cell">${formatCurrency(row.yhat_lower)}</td>
                        <td class="confidence-cell">${formatCurrency(row.yhat_upper)}</td>
                    </tr>
                `).join('')}
            </tbody>
        `;

        table.innerHTML = tableHTML;
    }

    loadInsightsSection() {
        this.loadFileList();
    }

    loadForecastSection() {
        this.loadFileList();
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new RetailIntelligenceApp();
});

// Ensure showSection is globally available
function showSection(sectionName) {
    if (window.app) {
        window.app.showSection(sectionName);
    }
}
window.showSection = showSection;