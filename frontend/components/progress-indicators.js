/**
 * Reusable progress indicator components for FoodVisionAI
 */

// Alpine.js component for circular progress indicator
function circularProgress() {
    return {
        progress: 0,
        size: 40,
        strokeWidth: 4,
        
        get circumference() {
            return 2 * Math.PI * (this.size / 2 - this.strokeWidth);
        },
        
        get strokeDasharray() {
            return this.circumference;
        },
        
        get strokeDashoffset() {
            return this.circumference - (this.progress / 100) * this.circumference;
        },
        
        get viewBox() {
            return `0 0 ${this.size} ${this.size}`;
        }
    }
}

// Alpine.js component for linear progress bar
function linearProgress() {
    return {
        progress: 0,
        animated: false,
        color: 'blue',
        
        get progressStyle() {
            return `width: ${this.progress}%; transition: width 0.5s ease-in-out;`;
        },
        
        get colorClasses() {
            const colors = {
                blue: 'bg-blue-600',
                green: 'bg-green-600',
                purple: 'bg-purple-600',
                red: 'bg-red-600',
                yellow: 'bg-yellow-600'
            };
            return colors[this.color] || colors.blue;
        }
    }
}

// Alpine.js component for step progress indicator
function stepProgress() {
    return {
        steps: [],
        currentStep: 0,
        
        init() {
            if (!this.steps.length) {
                this.steps = [
                    { name: 'Vision Analysis', status: 'pending' },
                    { name: 'Nutrition Analysis', status: 'pending' },
                    { name: 'Cuisine Classification', status: 'pending' }
                ];
            }
        },
        
        updateStep(index, status) {
            if (this.steps[index]) {
                this.steps[index].status = status;
                if (status === 'completed') {
                    this.currentStep = Math.max(this.currentStep, index + 1);
                }
            }
        },
        
        getStepClasses(index, status) {
            const baseClasses = 'flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium';
            
            switch (status) {
                case 'completed':
                    return `${baseClasses} bg-green-600 text-white`;
                case 'processing':
                    return `${baseClasses} bg-blue-600 text-white animate-pulse`;
                case 'failed':
                    return `${baseClasses} bg-red-600 text-white`;
                default:
                    return `${baseClasses} bg-gray-300 text-gray-600`;
            }
        },
        
        getConnectorClasses(index) {
            const nextStep = this.steps[index + 1];
            if (!nextStep) return '';
            
            const isCompleted = this.steps[index].status === 'completed';
            return isCompleted ? 'bg-green-600' : 'bg-gray-300';
        }
    }
}

// Alpine.js component for loading spinner
function loadingSpinner() {
    return {
        size: 'md',
        color: 'blue',
        
        get sizeClasses() {
            const sizes = {
                sm: 'w-4 h-4',
                md: 'w-8 h-8',
                lg: 'w-12 h-12',
                xl: 'w-16 h-16'
            };
            return sizes[this.size] || sizes.md;
        },
        
        get colorClasses() {
            const colors = {
                blue: 'text-blue-600',
                green: 'text-green-600',
                purple: 'text-purple-600',
                red: 'text-red-600',
                gray: 'text-gray-600'
            };
            return colors[this.color] || colors.blue;
        }
    }
}

// Alpine.js component for pulse animation
function pulseIndicator() {
    return {
        active: false,
        color: 'blue',
        
        get pulseClasses() {
            if (!this.active) return '';
            
            const colors = {
                blue: 'animate-pulse bg-blue-400',
                green: 'animate-pulse bg-green-400',
                purple: 'animate-pulse bg-purple-400',
                red: 'animate-pulse bg-red-400',
                yellow: 'animate-pulse bg-yellow-400'
            };
            return colors[this.color] || colors.blue;
        }
    }
}

// Alpine.js component for skeleton loader
function skeletonLoader() {
    return {
        lines: 3,
        animated: true,
        
        get animationClasses() {
            return this.animated ? 'animate-pulse' : '';
        }
    }
}

// Export components for use in Alpine.js
window.progressComponents = {
    circularProgress,
    linearProgress,
    stepProgress,
    loadingSpinner,
    pulseIndicator,
    skeletonLoader
};