/**
 * Centralized error handling for FoodVisionAI frontend
 */

class ErrorHandler {
    constructor() {
        this.errorContainer = null;
        this.retryCallbacks = new Map();
    }

    init(containerSelector = '#error-container') {
        this.errorContainer = document.querySelector(containerSelector);
        if (!this.errorContainer) {
            // Create error container if it doesn't exist
            this.createErrorContainer();
        }
    }

    createErrorContainer() {
        const container = document.createElement('div');
        container.id = 'error-container';
        container.className = 'fixed top-4 right-4 z-50 max-w-md';
        document.body.appendChild(container);
        this.errorContainer = container;
    }

    showError(message, options = {}) {
        const {
            type = 'error',
            duration = 5000,
            retryable = false,
            onRetry = null,
            persistent = false
        } = options;

        const errorId = `error-${Date.now()}`;
        const errorElement = this.createErrorElement(errorId, message, type, retryable, onRetry);
        
        this.errorContainer.appendChild(errorElement);

        if (onRetry) {
            this.retryCallbacks.set(errorId, onRetry);
        }

        // Auto-remove non-persistent errors
        if (!persistent && duration > 0) {
            setTimeout(() => {
                this.removeError(errorId);
            }, duration);
        }

        return errorId;
    }

    createErrorElement(id, message, type, retryable, onRetry) {
        const typeConfig = {
            error: {
                bgColor: 'bg-red-50',
                borderColor: 'border-red-200',
                textColor: 'text-red-800',
                iconColor: 'text-red-400',
                icon: 'M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z'
            },
            warning: {
                bgColor: 'bg-yellow-50',
                borderColor: 'border-yellow-200',
                textColor: 'text-yellow-800',
                iconColor: 'text-yellow-400',
                icon: 'M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z'
            },
            info: {
                bgColor: 'bg-blue-50',
                borderColor: 'border-blue-200',
                textColor: 'text-blue-800',
                iconColor: 'text-blue-400',
                icon: 'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z'
            }
        };

        const config = typeConfig[type] || typeConfig.error;

        const element = document.createElement('div');
        element.id = id;
        element.className = `${config.bgColor} ${config.borderColor} border rounded-lg p-4 mb-3 shadow-lg animate-slide-in`;
        
        element.innerHTML = `
            <div class="flex items-start">
                <svg class="w-5 h-5 ${config.iconColor} mt-0.5 mr-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="${config.icon}"/>
                </svg>
                <div class="flex-1">
                    <p class="${config.textColor} text-sm font-medium">${message}</p>
                    ${retryable ? `
                        <button 
                            onclick="errorHandler.retry('${id}')"
                            class="mt-2 text-xs ${config.textColor} underline hover:no-underline"
                        >
                            Try Again
                        </button>
                    ` : ''}
                </div>
                <button 
                    onclick="errorHandler.removeError('${id}')"
                    class="${config.iconColor} hover:${config.textColor} ml-2"
                >
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                    </svg>
                </button>
            </div>
        `;

        return element;
    }

    removeError(errorId) {
        const element = document.getElementById(errorId);
        if (element) {
            element.classList.add('animate-slide-out');
            setTimeout(() => {
                element.remove();
                this.retryCallbacks.delete(errorId);
            }, 300);
        }
    }

    retry(errorId) {
        const callback = this.retryCallbacks.get(errorId);
        if (callback && typeof callback === 'function') {
            callback();
            this.removeError(errorId);
        }
    }

    clearAll() {
        if (this.errorContainer) {
            this.errorContainer.innerHTML = '';
            this.retryCallbacks.clear();
        }
    }

    // Network error handling
    handleNetworkError(error, context = '') {
        let message = 'Network error occurred';
        
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            message = 'Unable to connect to server. Please check your internet connection.';
        } else if (error.status) {
            switch (error.status) {
                case 400:
                    message = 'Invalid request. Please check your input.';
                    break;
                case 401:
                    message = 'Authentication required. Please refresh the page.';
                    break;
                case 403:
                    message = 'Access denied. You don\'t have permission for this action.';
                    break;
                case 404:
                    message = 'Resource not found. The requested item may have been deleted.';
                    break;
                case 413:
                    message = 'File too large. Please select a smaller image.';
                    break;
                case 429:
                    message = 'Too many requests. Please wait a moment and try again.';
                    break;
                case 500:
                    message = 'Server error. Please try again later.';
                    break;
                case 503:
                    message = 'Service temporarily unavailable. Please try again later.';
                    break;
                default:
                    message = `Server error (${error.status}). Please try again.`;
            }
        }

        if (context) {
            message = `${context}: ${message}`;
        }

        return this.showError(message, {
            type: 'error',
            retryable: true,
            duration: 8000
        });
    }

    // File validation errors
    handleFileError(file, error) {
        let message = 'File validation failed';

        if (error.includes('type')) {
            message = `Invalid file type. Please select an image file (JPEG, PNG, WebP, etc.)`;
        } else if (error.includes('size')) {
            message = `File too large (${this.formatFileSize(file.size)}). Maximum size is 10MB.`;
        } else if (error.includes('corrupt')) {
            message = 'File appears to be corrupted. Please try a different image.';
        }

        return this.showError(message, {
            type: 'error',
            duration: 6000
        });
    }

    formatFileSize(bytes) {
        if (!bytes) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Success messages
    showSuccess(message, duration = 4000) {
        return this.showError(message, {
            type: 'info',
            duration: duration
        });
    }

    // Warning messages
    showWarning(message, duration = 5000) {
        return this.showError(message, {
            type: 'warning',
            duration: duration
        });
    }
}

// Global error handler instance
const errorHandler = new ErrorHandler();

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => errorHandler.init());
} else {
    errorHandler.init();
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slide-in {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slide-out {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .animate-slide-in {
        animation: slide-in 0.3s ease-out;
    }
    
    .animate-slide-out {
        animation: slide-out 0.3s ease-in;
    }
`;
document.head.appendChild(style);

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ErrorHandler;
}