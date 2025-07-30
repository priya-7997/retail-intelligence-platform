 /**
 * Utility Functions for Retail Intelligence Platform
 * Common helper functions used across the application
 */

// Currency formatting for Indian Rupees
function formatCurrency(amount) {
    if (typeof amount !== 'number') {
        return '₹0';
    }
    
    if (amount >= 10000000) { // 1 Crore
        return `₹${(amount / 10000000).toFixed(2)}Cr`;
    } else if (amount >= 100000) { // 1 Lakh
        return `₹${(amount / 100000).toFixed(2)}L`;
    } else if (amount >= 1000) { // 1 Thousand
        return `₹${(amount / 1000).toFixed(2)}K`;
    } else {
        return `₹${amount.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }
}

// Number formatting for Indian numbering system
function formatNumber(number) {
    if (typeof number !== 'number') {
        return '0';
    }
    
    if (number >= 10000000) { // 1 Crore
        return `${(number / 10000000).toFixed(2)}Cr`;
    } else if (number >= 100000) { // 1 Lakh
        return `${(number / 100000).toFixed(2)}L`;
    } else if (number >= 1000) { // 1 Thousand
        return `${(number / 1000).toFixed(2)}K`;
    } else {
        return number.toLocaleString('en-IN');
    }
}

// Date formatting
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-IN', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

// Relative time formatting
function formatRelativeTime(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} min ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    
    return formatDate(dateString);
}

// Percentage formatting
function formatPercentage(value, decimals = 1) {
    if (typeof value !== 'number') {
        return '0%';
    }
    return `${value.toFixed(decimals)}%`;
}

// File size formatting
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Notification system
class NotificationManager {
    constructor() {
        this.container = document.getElementById('notifications');
        this.notifications = new Map();
        this.counter = 0;
    }
    
    show(message, type = 'info', duration = 5000, persistent = false) {
        const id = `notification-${++this.counter}`;
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.id = id;
        
        notification.innerHTML = `
            <div class="notification-content">
                <div class="notification-icon">
                    ${this.getIcon(type)}
                </div>
                <div class="notification-message">${message}</div>
                <button class="notification-close" onclick="notifications.hide('${id}')">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        this.container.appendChild(notification);
        this.notifications.set(id, notification);
        
        // Auto-hide after duration (unless persistent)
        if (!persistent && duration > 0) {
            setTimeout(() => this.hide(id), duration);
        }
        
        return id;
    }
    
    hide(id) {
        const notification = this.notifications.get(id);
        if (notification) {
            notification.style.animation = 'slideOutRight 0.3s ease-out forwards';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
                this.notifications.delete(id);
            }, 300);
        }
    }
    
    clear() {
        this.notifications.forEach((notification, id) => {
            this.hide(id);
        });
    }
    
    getIcon(type) {
        const icons = {
            success: '<i class="fas fa-check-circle"></i>',
            error: '<i class="fas fa-exclamation-circle"></i>',
            warning: '<i class="fas fa-exclamation-triangle"></i>',
            info: '<i class="fas fa-info-circle"></i>'
        };
        return icons[type] || icons.info;
    }
}

// Initialize global notification manager
const notifications = new NotificationManager();

// Loading overlay utilities
function showLoading(title = 'Processing...', message = 'Please wait while we process your request') {
    const overlay = document.getElementById('loadingOverlay');
    const titleEl = document.getElementById('loadingTitle');
    const messageEl = document.getElementById('loadingMessage');
    
    if (titleEl) titleEl.textContent = title;
    if (messageEl) messageEl.textContent = message;
    
    overlay.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.style.display = 'none';
    document.body.style.overflow = '';
}

// Debounce function for search and input handling
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func(...args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func(...args);
    };
}

// Throttle function for scroll and resize events
function throttle(func, limit) {
    let lastFunc;
    let lastRan;
    return function() {
        const context = this;
        const args = arguments;
        if (!lastRan) {
            func.apply(context, args);
            lastRan = Date.now();
        } else {
            clearTimeout(lastFunc);
            lastFunc = setTimeout(function() {
                if ((Date.now() - lastRan) >= limit) {
                    func.apply(context, args);
                    lastRan = Date.now();
                }
            }, limit - (Date.now() - lastRan));
        }
    };
}

// DOM utility functions
function createElement(tag, className = '', innerHTML = '') {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (innerHTML) element.innerHTML = innerHTML;
    return element;
}

function removeElement(element) {
    if (element && element.parentNode) {
        element.parentNode.removeChild(element);
    }
}

function show(element) {
    if (element) {
        element.style.display = '';
        element.classList.remove('hidden');
    }
}

function hide(element) {
    if (element) {
        element.style.display = 'none';
        element.classList.add('hidden');
    }
}

function toggle(element) {
    if (element) {
        if (element.style.display === 'none' || element.classList.contains('hidden')) {
            show(element);
        } else {
            hide(element);
        }
    }
}

// Animation utilities
function fadeIn(element, duration = 300) {
    if (!element) return;
    
    element.style.opacity = '0';
    element.style.display = '';
    element.classList.remove('hidden');
    
    let start = null;
    function animate(timestamp) {
        if (!start) start = timestamp;
        const progress = timestamp - start;
        
        element.style.opacity = Math.min(progress / duration, 1);
        
        if (progress < duration) {
            requestAnimationFrame(animate);
        }
    }
    
    requestAnimationFrame(animate);
}

function fadeOut(element, duration = 300) {
    if (!element) return;
    
    let start = null;
    const initialOpacity = parseFloat(getComputedStyle(element).opacity) || 1;
    
    function animate(timestamp) {
        if (!start) start = timestamp;
        const progress = timestamp - start;
        
        element.style.opacity = initialOpacity * (1 - progress / duration);
        
        if (progress < duration) {
            requestAnimationFrame(animate);
        } else {
            hide(element);
        }
    }
    
    requestAnimationFrame(animate);
}

// Local storage utilities with error handling
function setStorage(key, value) {
    try {
        localStorage.setItem(key, JSON.stringify(value));
        return true;
    } catch (e) {
        console.warn('Failed to save to localStorage:', e);
        return false;
    }
}

function getStorage(key, defaultValue = null) {
    try {
        const item = localStorage.getItem(key);
        return item ? JSON.parse(item) : defaultValue;
    } catch (e) {
        console.warn('Failed to read from localStorage:', e);
        return defaultValue;
    }
}

function removeStorage(key) {
    try {
        localStorage.removeItem(key);
        return true;
    } catch (e) {
        console.warn('Failed to remove from localStorage:', e);
        return false;
    }
}

// Form validation utilities
function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

function validateRequired(value) {
    return value !== null && value !== undefined && value.toString().trim() !== '';
}

function validateNumber(value, min = null, max = null) {
    const num = parseFloat(value);
    if (isNaN(num)) return false;
    if (min !== null && num < min) return false;
    if (max !== null && num > max) return false;
    return true;
}

// CSV export utility
function exportToCSV(data, filename = 'export.csv') {
    if (!data || !data.length) {
        notifications.show('No data to export', 'warning');
        return;
    }
    
    // Get headers from first object
    const headers = Object.keys(data[0]);
    
    // Create CSV content
    let csvContent = headers.join(',') + '\n';
    
    data.forEach(row => {
        const values = headers.map(header => {
            let value = row[header] || '';
            // Escape quotes and wrap in quotes if contains comma
            if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
                value = '"' + value.replace(/"/g, '""') + '"';
            }
            return value;
        });
        csvContent += values.join(',') + '\n';
    });
    
    // Download file
    downloadFile(csvContent, filename, 'text/csv');
}

// File download utility
function downloadFile(content, filename, contentType = 'text/plain') {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    URL.revokeObjectURL(url);
}

// URL parameter utilities
function getUrlParameter(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}

// URL parameter utilities
function getUrlParameter(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}

function setUrlParameter(name, value) {
    const url = new URL(window.location);
    url.searchParams.set(name, value);
    window.history.pushState({}, '', url);
}

function removeUrlParameter(name) {
    const url = new URL(window.location);
    url.searchParams.delete(name);
    window.history.pushState({}, '', url);
}

// Copy to clipboard utility
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        notifications.show('Copied to clipboard', 'success', 2000);
        return true;
    } catch (err) {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        
        try {
            document.execCommand('copy');
            notifications.show('Copied to clipboard', 'success', 2000);
            return true;
        } catch (fallbackErr) {
            notifications.show('Failed to copy to clipboard', 'error');
            return false;
        } finally {
            document.body.removeChild(textArea);
        }
    }
}

// Color utilities for charts
const chartColors = {
    primary: '#3b82f6',
    secondary: '#10b981',
    accent: '#f59e0b',
    error: '#ef4444',
    warning: '#f59e0b',
    success: '#10b981',
    info: '#3b82f6',
    gray: '#6b7280'
};

function getChartColorPalette(count = 10) {
    const baseColors = [
        '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
        '#06b6d4', '#84cc16', '#f97316', '#ec4899', '#64748b'
    ];
    
    if (count <= baseColors.length) {
        return baseColors.slice(0, count);
    }
    
    // Generate additional colors if needed
    const colors = [...baseColors];
    while (colors.length < count) {
        const hue = (colors.length * 137.508) % 360; // Golden angle
        colors.push(`hsl(${hue}, 65%, 55%)`);
    }
    
    return colors;
}

// Data transformation utilities
function groupBy(array, key) {
    return array.reduce((groups, item) => {
        const group = item[key];
        if (!groups[group]) {
            groups[group] = [];
        }
        groups[group].push(item);
        return groups;
    }, {});
}

function sortBy(array, key, ascending = true) {
    return [...array].sort((a, b) => {
        const aVal = a[key];
        const bVal = b[key];
        
        if (aVal < bVal) return ascending ? -1 : 1;
        if (aVal > bVal) return ascending ? 1 : -1;
        return 0;
    });
}

function sumBy(array, key) {
    return array.reduce((sum, item) => sum + (item[key] || 0), 0);
}

function averageBy(array, key) {
    if (!array.length) return 0;
    return sumBy(array, key) / array.length;
}

function uniqueBy(array, key) {
    const seen = new Set();
    return array.filter(item => {
        const value = item[key];
        if (seen.has(value)) {
            return false;
        }
        seen.add(value);
        return true;
    });
}

// Table utilities
function createTable(data, columns, options = {}) {
    if (!data || !data.length) {
        return '<div class="no-data">No data available</div>';
    }
    
    const {
        className = 'data-table',
        sortable = false,
        pagination = false,
        pageSize = 10
    } = options;
    
    let html = `<table class="${className}">`;
    
    // Header
    html += '<thead><tr>';
    columns.forEach(col => {
        const sortClass = sortable ? 'sortable' : '';
        html += `<th class="${sortClass}" data-key="${col.key}">${col.label}</th>`;
    });
    html += '</tr></thead>';
    
    // Body
    html += '<tbody>';
    data.forEach(row => {
        html += '<tr>';
        columns.forEach(col => {
            let value = row[col.key];
            if (col.formatter) {
                value = col.formatter(value, row);
            }
            html += `<td>${value}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    
    return html;
}

// Error handling utilities
function handleError(error, context = '') {
    console.error(`Error ${context}:`, error);
    
    let message = 'An unexpected error occurred';
    
    if (error.message) {
        message = error.message;
    } else if (typeof error === 'string') {
        message = error;
    }
    
    notifications.show(message, 'error', 8000);
    
    // Log to analytics/monitoring service if available
    if (window.analytics && window.analytics.track) {
        window.analytics.track('Error', {
            message: message,
            context: context,
            stack: error.stack
        });
    }
}

// Performance monitoring utilities
class PerformanceMonitor {
    constructor() {
        this.marks = new Map();
        this.measures = new Map();
    }
    
    start(name) {
        this.marks.set(name, performance.now());
    }
    
    end(name) {
        const startTime = this.marks.get(name);
        if (startTime) {
            const duration = performance.now() - startTime;
            this.measures.set(name, duration);
            console.log(`⏱️ ${name}: ${duration.toFixed(2)}ms`);
            return duration;
        }
        return null;
    }
    
    getMeasure(name) {
        return this.measures.get(name);
    }
    
    getAllMeasures() {
        return Object.fromEntries(this.measures);
    }
    
    clear() {
        this.marks.clear();
        this.measures.clear();
    }
}

// Initialize global performance monitor
const perfMonitor = new PerformanceMonitor();

// Browser compatibility checks
function checkBrowserSupport() {
    const features = {
        fetch: typeof fetch !== 'undefined',
        promises: typeof Promise !== 'undefined',
        localStorage: typeof Storage !== 'undefined',
        canvas: !!document.createElement('canvas').getContext,
        fileAPI: !!(window.File && window.FileReader && window.FileList && window.Blob)
    };
    
    const unsupported = Object.entries(features)
        .filter(([name, supported]) => !supported)
        .map(([name]) => name);
    
    if (unsupported.length > 0) {
        console.warn('Unsupported browser features:', unsupported);
        notifications.show(
            'Your browser may not support all features. Please update to the latest version.',
            'warning',
            10000
        );
    }
    
    return features;
}

// Initialize browser check on load
document.addEventListener('DOMContentLoaded', checkBrowserSupport);

// Math utilities for calculations
function calculateGrowthRate(current, previous) {
    if (!previous || previous === 0) return 0;
    return ((current - previous) / previous) * 100;
}

function calculateMovingAverage(data, window = 7) {
    if (!data || data.length < window) return data;
    
    const result = [];
    for (let i = 0; i < data.length; i++) {
        if (i < window - 1) {
            result.push(null);
        } else {
            const sum = data.slice(i - window + 1, i + 1).reduce((a, b) => a + b, 0);
            result.push(sum / window);
        }
    }
    return result;
}

function calculatePercentile(data, percentile) {
    if (!data || !data.length) return 0;
    
    const sorted = [...data].sort((a, b) => a - b);
    const index = (percentile / 100) * (sorted.length - 1);
    
    if (Math.floor(index) === index) {
        return sorted[index];
    } else {
        const lower = sorted[Math.floor(index)];
        const upper = sorted[Math.ceil(index)];
        const weight = index - Math.floor(index);
        return lower * (1 - weight) + upper * weight;
    }
}

function calculateStandardDeviation(data) {
    if (!data || !data.length) return 0;
    
    const mean = data.reduce((sum, value) => sum + value, 0) / data.length;
    const squaredDiffs = data.map(value => Math.pow(value - mean, 2));
    const avgSquaredDiff = squaredDiffs.reduce((sum, value) => sum + value, 0) / data.length;
    
    return Math.sqrt(avgSquaredDiff);
}

// String utilities
function truncateText(text, maxLength = 100, suffix = '...') {
    if (!text || text.length <= maxLength) return text;
    return text.substring(0, maxLength - suffix.length) + suffix;
}

function slugify(text) {
    return text
        .toLowerCase()
        .trim()
        .replace(/[^\w\s-]/g, '')
        .replace(/[\s_-]+/g, '-')
        .replace(/^-+|-+$/g, '');
}

function capitalizeFirst(text) {
    if (!text) return '';
    return text.charAt(0).toUpperCase() + text.slice(1);
}

function titleCase(text) {
    if (!text) return '';
    return text.replace(/\w\S*/g, (txt) => 
        txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase()
    );
}

// Date utilities
function addDays(date, days) {
    const result = new Date(date);
    result.setDate(result.getDate() + days);
    return result;
}

function addMonths(date, months) {
    const result = new Date(date);
    result.setMonth(result.getMonth() + months);
    return result;
}

function getDaysDifference(date1, date2) {
    const timeDiff = Math.abs(date2.getTime() - date1.getTime());
    return Math.ceil(timeDiff / (1000 * 3600 * 24));
}

function isWeekend(date) {
    const day = date.getDay();
    return day === 0 || day === 6; // Sunday = 0, Saturday = 6
}

function getQuarter(date) {
    return Math.floor((date.getMonth() + 3) / 3);
}

// Export utilities for global use
window.utils = {
    formatCurrency,
    formatNumber,
    formatDate,
    formatRelativeTime,
    formatPercentage,
    formatFileSize,
    debounce,
    throttle,
    createElement,
    removeElement,
    show,
    hide,
    toggle,
    fadeIn,
    fadeOut,
    setStorage,
    getStorage,
    removeStorage,
    validateEmail,
    validateRequired,
    validateNumber,
    exportToCSV,
    downloadFile,
    getUrlParameter,
    setUrlParameter,
    removeUrlParameter,
    copyToClipboard,
    chartColors,
    getChartColorPalette,
    groupBy,
    sortBy,
    sumBy,
    averageBy,
    uniqueBy,
    createTable,
    handleError,
    calculateGrowthRate,
    calculateMovingAverage,
    calculatePercentile,
    calculateStandardDeviation,
    truncateText,
    slugify,
    capitalizeFirst,
    titleCase,
    addDays,
    addMonths,
    getDaysDifference,
    isWeekend,
    getQuarter
};

// Export notification manager and performance monitor
window.notifications = notifications;
window.perfMonitor = perfMonitor;
