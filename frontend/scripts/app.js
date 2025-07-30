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
                this.showSection(section);
            });
        });

        // File upload
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');

        if (fileInput && uploadArea) {
            uploadArea.addEventListener('click', () => fileInput.click());
            fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
            
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
                if (files.length > 0) {
                    this.uploadFile(files[0]);
                }
            });
        }

        // Generate buttons
        const generateInsightsBtn = document.getElementById('generateInsights');
        const generateForecastBtn = document.getElementById('generateForecast');

        if (generateInsightsBtn) {
            generateInsightsBtn.addEventListener('click', () => this.generateInsights());
        }

        if (generateForecastBtn) {
            generateForecastBtn.addEventListener('click', () => this.generateForecast());
        }
    }

    showSection(sectionName) {
        // Update navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
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
            const result = await uploadAPI.listFiles();
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
            
            const result = await dashboardAPI.getOverview(fileId);
            
            if (result.success) {
                this.displayDashboard(result.dashboard);
                hide(document.getElementById('welcome-state'));
            }
            
        } catch (error) {
            console.error('Dashboard loading error:', error);
            notifications.show('Failed to load dashboard', 'error');
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
                <div class="insight-preview-content">${insight.content}</div>
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

        show(document.getElementById('insights-dashboard'));
    }

    displayExecutiveSummary(summary) {
        const container = document.getElementById('executiveSummary');
        if (!container || !summary) return;

        container.innerHTML = summary.map(item => `
            <div class="insight-item">
                <i class="fas fa-chart-line"></i>
                <span>${item}</span>
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
            { id: 'salesInsights', key: 'sales_insights', icon: 'trending-up' },
            { id: 'seasonalInsights', key: 'seasonality_insights', icon: 'calendar-alt' },
            { id: 'inventoryInsights', key: 'inventory_alerts', icon: 'boxes' },
            { id: 'revenueInsights', key: 'revenue_opportunities', icon: 'dollar-sign' }
        ];

        categories.forEach(category => {
            const container = document.getElementById(category.id);
            const data = insights[category.key];
            
            if (container && data) {
                container.innerHTML = data.map(item => `
                    <div class="insight-item">
                        <i class="fas fa-${category.icon}"></i>
                        <span>${item}</span>
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
        
        const fileId = fileSelector?.value;
        const periods = parseInt(periodsSelector?.value || '30');
        const modelType = modelSelector?.value || 'auto';
        
        if (!fileId) {
            notifications.show('Please select a processed file', 'warning');
            return;
        }

        try {
            show(document.getElementById('forecast-loading'));
            hide(document.getElementById('forecast-results'));
            
            let result;
            if (modelType === 'auto') {
                result = await generateAutoForecast(fileId, periods);
            } else {
                result = await forecastAPI.generateForecast({
                    file_id: fileId,
                    model_type: modelType,
                    forecast_periods: periods
                });
            }
            
            if (result.success) {
                this.displayForecastResults(result);
            }
            
        } catch (error) {
            console.error('Forecast generation error:', error);
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

        chartManager.createLineChart('forecastChart', data, {
            title: 'Sales Forecast with Confidence Intervals'
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

// Global function to show sections (for button clicks)
function showSection(sectionName) {
    if (window.app) {
        window.app.showSection(sectionName);
    }
}