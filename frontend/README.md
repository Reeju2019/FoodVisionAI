# FoodVisionAI Frontend

Modern, responsive frontend for the FoodVisionAI application built with HTML, Alpine.js, and Tailwind CSS.

## Features

### 🎨 Modern UI/UX
- **Responsive Design**: Works seamlessly across desktop, tablet, and mobile devices
- **Tailwind CSS**: Utility-first CSS framework for rapid UI development
- **Alpine.js**: Lightweight JavaScript framework for reactive components
- **Gradient Backgrounds**: Beautiful gradient backgrounds for visual appeal
- **Card-based Layout**: Clean, modern card-based interface design

### 📤 Upload Page (`upload.html`)
- **Drag & Drop**: Intuitive drag-and-drop file upload interface
- **File Picker**: Traditional file picker as fallback option
- **Real-time Validation**: Client-side file type and size validation
- **Image Preview**: Live preview of selected images before upload
- **Progress Indicators**: Visual upload progress with animated states
- **Error Handling**: Comprehensive error messages with retry options

### 📊 Analytics Page (`analytics.html`)
- **Real-time Updates**: Smart polling with exponential backoff
- **Progress Tracking**: Visual progress bars for each AI model stage
- **Results Display**: Organized display of vision, nutrition, and cuisine results
- **Status Indicators**: Live status updates with color-coded indicators
- **Error Recovery**: Graceful error handling with retry mechanisms
- **Responsive Grid**: Adaptive layout for different screen sizes

### 🔧 Components

#### Progress Indicators (`components/progress-indicators.js`)
- **Circular Progress**: Animated circular progress indicators
- **Linear Progress**: Horizontal progress bars with smooth animations
- **Step Progress**: Multi-step progress visualization
- **Loading Spinners**: Various loading spinner styles
- **Pulse Indicators**: Animated pulse indicators for active states

#### Loading States (`components/loading-states.html`)
- **Skeleton Loaders**: Placeholder content while loading
- **Analysis Stages**: Visual representation of AI pipeline stages
- **Error States**: Comprehensive error state components
- **Success States**: Success message components

#### Error Handler (`js/error-handler.js`)
- **Centralized Error Management**: Global error handling system
- **Network Error Detection**: Automatic network error classification
- **File Validation Errors**: Specialized file validation error messages
- **Retry Mechanisms**: Built-in retry functionality for failed operations
- **Toast Notifications**: Slide-in notification system

## Technical Implementation

### Alpine.js Integration
```javascript
// Example component structure
function uploadApp() {
    return {
        selectedFile: null,
        isUploading: false,
        uploadProgress: 0,
        
        async uploadFile() {
            // Upload logic with progress tracking
        },
        
        handleFile(file) {
            // File validation and preview
        }
    }
}
```

### Real-time Polling
```javascript
// Smart polling with exponential backoff
scheduleNextPoll() {
    const delay = this.baseDelay * Math.pow(1.5, Math.min(this.retryCount, 5));
    setTimeout(() => this.fetchStatus(), delay);
}
```

### Error Handling
```javascript
// Centralized error management
errorHandler.handleNetworkError(error, 'Upload failed');
errorHandler.showSuccess('Upload completed successfully!');
```

## File Structure

```
frontend/
├── upload.html                 # Main upload page
├── analytics.html              # Analysis results page
├── js/
│   └── error-handler.js        # Global error handling
├── components/
│   ├── progress-indicators.js  # Reusable progress components
│   └── loading-states.html     # Loading state templates
└── README.md                   # This file
```

## API Integration

### Upload Endpoint
```javascript
POST /api/v1/upload
Content-Type: multipart/form-data

// Response
{
    "image_id": "uuid",
    "status": "uploaded",
    "analytics_url": "/api/v1/analytics/{image_id}"
}
```

### Status Polling
```javascript
GET /api/v1/status/{image_id}

// Response
{
    "status": "processing|completed|failed",
    "progress": {
        "vision": {"completed": true, "progress": 1.0},
        "nutrition": {"completed": false, "progress": 0.5},
        "cuisine": {"completed": false, "progress": 0.0}
    },
    "results": {
        "vision": {...},
        "nutrition": {...},
        "cuisine": {...}
    }
}
```

## Validation & Error Handling

### Client-side Validation
- **File Type**: Only image files (JPEG, PNG, WebP, etc.)
- **File Size**: Maximum 10MB file size limit
- **File Integrity**: Basic file corruption detection

### Error Categories
1. **Network Errors**: Connection issues, timeouts
2. **Validation Errors**: Invalid file types, size limits
3. **Server Errors**: HTTP 4xx/5xx responses
4. **Processing Errors**: AI model failures

### Recovery Mechanisms
- **Automatic Retry**: Network errors with exponential backoff
- **Manual Retry**: User-initiated retry for failed operations
- **Graceful Degradation**: Partial results display when possible

## Performance Optimizations

### Smart Polling
- **Exponential Backoff**: Reduces server load during long processing
- **Conditional Polling**: Stops polling when analysis completes
- **Error Throttling**: Prevents excessive requests on errors

### Resource Management
- **Image Optimization**: Client-side image preview optimization
- **Memory Management**: Proper cleanup of file objects and intervals
- **Lazy Loading**: Components loaded only when needed

### Caching Strategy
- **Static Assets**: CDN delivery for Alpine.js and Tailwind CSS
- **Component Reuse**: Reusable components to reduce code duplication
- **State Management**: Efficient state updates to minimize re-renders

## Browser Compatibility

### Supported Browsers
- **Chrome**: 80+
- **Firefox**: 75+
- **Safari**: 13+
- **Edge**: 80+

### Progressive Enhancement
- **Core Functionality**: Works without JavaScript (basic forms)
- **Enhanced Experience**: Full features with JavaScript enabled
- **Responsive Design**: Mobile-first approach with desktop enhancements

## Development Guidelines

### Code Style
- **Consistent Naming**: camelCase for JavaScript, kebab-case for CSS
- **Component Structure**: Modular, reusable components
- **Error Handling**: Comprehensive error handling at all levels
- **Documentation**: Inline comments for complex logic

### Testing Considerations
- **File Upload Testing**: Various file types and sizes
- **Network Testing**: Offline/online scenarios
- **Error Testing**: Simulated error conditions
- **Cross-browser Testing**: Multiple browser environments

## Future Enhancements

### Planned Features
- **Offline Support**: Service worker for offline functionality
- **Image Editing**: Basic image editing before upload
- **Batch Upload**: Multiple image upload support
- **Export Options**: PDF/CSV export of analysis results
- **User Preferences**: Customizable UI themes and settings

### Performance Improvements
- **Image Compression**: Client-side image compression
- **WebP Support**: Modern image format support
- **Lazy Loading**: Progressive image loading
- **Caching**: Advanced caching strategies

## Deployment

### Static File Serving
The FastAPI backend serves frontend files through:
```python
app.mount("/static", StaticFiles(directory="frontend"), name="static")
```

### CDN Integration
External resources loaded from CDN:
- **Alpine.js**: `https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js`
- **Tailwind CSS**: `https://cdn.tailwindcss.com`

### Production Considerations
- **Minification**: Minify JavaScript and CSS for production
- **Compression**: Enable gzip compression for static assets
- **Security**: Content Security Policy (CSP) headers
- **Monitoring**: Error tracking and performance monitoring