// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const textToggle = document.getElementById('text-toggle');
const fileToggle = document.getElementById('file-toggle');
const textForm = document.getElementById('text-form');
const fileForm = document.getElementById('file-form');
const textVerificationForm = document.getElementById('text-verification-form');
const fileVerificationForm = document.getElementById('file-verification-form');
const fileUploadArea = document.getElementById('file-upload-area');
const fileInput = document.getElementById('advertisement-file');
const fileInfo = document.getElementById('file-info');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('results-section');

// State Management
let currentFile = null;
let isProcessing = false;

// Initialize Application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    setupFileUpload();
});

// Event Listeners
function initializeEventListeners() {
    // Toggle between text and file input
    textToggle.addEventListener('click', () => switchInputMode('text'));
    fileToggle.addEventListener('click', () => switchInputMode('file'));
    
    // Form submissions
    textVerificationForm.addEventListener('submit', handleTextSubmission);
    fileVerificationForm.addEventListener('submit', handleFileSubmission);
}

// Switch Input Mode
function switchInputMode(mode) {
    if (isProcessing) return;
    
    if (mode === 'text') {
        textToggle.classList.add('active');
        fileToggle.classList.remove('active');
        textForm.classList.remove('hidden');
        fileForm.classList.add('hidden');
    } else {
        fileToggle.classList.add('active');
        textToggle.classList.remove('active');
        fileForm.classList.remove('hidden');
        textForm.classList.add('hidden');
    }
    
    // Hide results when switching modes
    hideResults();
}

// File Upload Setup
function setupFileUpload() {
    // Click to upload
    fileUploadArea.addEventListener('click', () => {
        if (!isProcessing) {
            fileInput.click();
        }
    });
    
    // File selection
    fileInput.addEventListener('change', handleFileSelection);
    
    // Drag and drop
    fileUploadArea.addEventListener('dragover', handleDragOver);
    fileUploadArea.addEventListener('dragleave', handleDragLeave);
    fileUploadArea.addEventListener('drop', handleDrop);
}

// Handle File Selection
function handleFileSelection(event) {
    const files = event.target.files;
    if (files.length > 0) {
        currentFile = files[0];
        displayFileInfo(currentFile);
    }
}

// Display File Information
function displayFileInfo(file) {
    const fileSize = (file.size / 1024).toFixed(2);
    fileInfo.innerHTML = `
        <div class="file-details">
            <i class="fas fa-file"></i>
            <span><strong>${file.name}</strong></span>
            <span>${fileSize} KB</span>
        </div>
    `;
    fileInfo.classList.add('show');
}

// Drag and Drop Handlers
function handleDragOver(event) {
    event.preventDefault();
    fileUploadArea.classList.add('drag-over');
}

function handleDragLeave(event) {
    event.preventDefault();
    fileUploadArea.classList.remove('drag-over');
}

function handleDrop(event) {
    event.preventDefault();
    fileUploadArea.classList.remove('drag-over');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        currentFile = files[0];
        fileInput.files = files;
        displayFileInfo(currentFile);
    }
}

// Handle Text Form Submission
async function handleTextSubmission(event) {
    event.preventDefault();
    
    if (isProcessing) return;
    
    const formData = new FormData(event.target);
    const text = formData.get('text').trim();
    
    if (!text) {
        showError('Please enter some text to verify.');
        return;
    }
    
    try {
        isProcessing = true;
        showLoading();
        
        const response = await fetch(`${API_BASE_URL}/verify-text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        displayResults(result);
        
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to verify text. Please try again.');
    } finally {
        isProcessing = false;
        hideLoading();
    }
}

// Handle File Form Submission
async function handleFileSubmission(event) {
    event.preventDefault();
    
    if (isProcessing) return;
    
    if (!currentFile) {
        showError('Please select a file to verify.');
        return;
    }
    
    try {
        isProcessing = true;
        showLoading();
        
        const formData = new FormData();
        formData.append('file', currentFile);
        
        const response = await fetch(`${API_BASE_URL}/verify-file`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        displayResults(result);
        
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to verify file. Please try again.');
    } finally {
        isProcessing = false;
        hideLoading();
    }
}

// Display Results
function displayResults(data) {
    hideLoading();
    
    // Original Text
    document.getElementById('original-text').textContent = data.original_text;
    
    // Verification Status
    const statusElement = document.getElementById('verification-status');
    statusElement.textContent = formatVerificationStatus(data.verification_result);
    statusElement.className = `verification-status ${data.verification_result.replace('_', '-')}`;
    
    // Credibility Score
    displayCredibilityScore(data.credibility_score, data.sentiment_analysis);
    
    // Sentiment Analysis
    displaySentimentAnalysis(data.sentiment_analysis);
    
    // Fact Check Details
    document.getElementById('fact-check-details').textContent = data.fact_check_details;
    
    // Recommendations
    displayRecommendations(data.recommendations);
    
    // Show results section
    resultsSection.classList.add('show');
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Format Verification Status
function formatVerificationStatus(status) {
    const statusMap = {
        'verified': '✅ Verified',
        'partially_verified': '⚠️ Partially Verified',
        'unverified': '❌ Unverified',
        'misleading': '🚫 Misleading',
        'analysis_completed': '🔍 Analysis Completed'
    };
    
    return statusMap[status] || status.charAt(0).toUpperCase() + status.slice(1);
}

// Display Credibility Score
function displayCredibilityScore(score, sentimentData) {
    const percentage = Math.round(score * 100);
    const scoreElement = document.getElementById('score-value');
    const circleElement = document.getElementById('score-circle');
    const descriptionElement = document.getElementById('score-description');
    
    // Update score value
    scoreElement.textContent = `${percentage}%`;
    
    // Update circle gradient with color based on score
    const degrees = (percentage / 100) * 360;
    let circleColor = '#667eea'; // Default blue
    
    if (percentage >= 80) {
        circleColor = '#48bb78'; // Green
    } else if (percentage >= 60) {
        circleColor = '#ed8936'; // Orange
    } else if (percentage >= 40) {
        circleColor = '#f56565'; // Red
    } else {
        circleColor = '#e53e3e'; // Dark red
    }
    
    circleElement.style.background = `conic-gradient(${circleColor} ${degrees}deg, #e2e8f0 ${degrees}deg)`;
    
    // Add bullshit image for very low credibility scores
    addBullshitImageIfNeeded(percentage);
    
    // Enhanced description with reliability info
    let description = '';
    let color = '';
    
    // Use enhanced credibility data if available
    if (sentimentData && sentimentData.credibility_level && sentimentData.reliability) {
        description = `${sentimentData.credibility_level} credibility - ${sentimentData.reliability}`;
        
        // Show score breakdown if available
        if (sentimentData.score_breakdown) {
            const breakdown = sentimentData.score_breakdown;
            description += `\n\nScore Breakdown:`;
            description += `\n• Base Score: ${breakdown.base_score}`;
            description += `\n• Language Analysis: ${breakdown.language_analysis > 0 ? '+' : ''}${breakdown.language_analysis}`;
            description += `\n• Sentiment Factor: ${breakdown.sentiment_factor > 0 ? '+' : ''}${breakdown.sentiment_factor}`;
            description += `\n• Red Flag Penalties: ${breakdown.red_flag_penalties}`;
            description += `\n• AI Analysis: ${breakdown.gemini_adjustment > 0 ? '+' : ''}${breakdown.gemini_adjustment}`;
            description += `\n• Factual Indicators: ${breakdown.factual_indicators > 0 ? '+' : ''}${breakdown.factual_indicators}`;
        }
    } else {
        // Fallback to original descriptions
        if (percentage >= 80) {
            description = 'High credibility - Claims appear to be well-supported and trustworthy.';
        } else if (percentage >= 60) {
            description = 'Moderate credibility - Some claims may need additional verification.';
        } else if (percentage >= 40) {
            description = 'Low credibility - Several claims appear questionable or unsupported.';
        } else {
            description = 'Very low credibility - Most claims appear to be misleading or false.';
        }
    }
    
    // Set color based on score
    if (percentage >= 80) {
        color = '#22543d';
    } else if (percentage >= 60) {
        color = '#744210';
    } else if (percentage >= 40) {
        color = '#c53030';
    } else {
        color = '#742a2a';
    }
    
    descriptionElement.textContent = description;
    descriptionElement.style.color = color;
    descriptionElement.style.whiteSpace = 'pre-line'; // Allow line breaks
}

// Add bullshit image for very low credibility scores
function addBullshitImageIfNeeded(percentage) {
    // Remove any existing bullshit image
    const existingImage = document.getElementById('bullshit-image');
    if (existingImage) {
        existingImage.remove();
    }
    
    // Add bullshit image if credibility score is very low (30% or below)
    if (percentage <= 30) {
        const credibilityScoreContainer = document.querySelector('.credibility-score');
        
        // Create the bullshit image element
        const bullshitImage = document.createElement('img');
        bullshitImage.id = 'bullshit-image';
        bullshitImage.src = 'bullshit.jpg';
        bullshitImage.alt = 'Bullshit Detected!';
        bullshitImage.title = 'This claim appears to be complete bullshit!';
        bullshitImage.className = 'bullshit-indicator';
        
        // Handle image load error gracefully
        bullshitImage.onerror = function() {
            // If image doesn't exist, create a text indicator instead
            const textIndicator = document.createElement('div');
            textIndicator.id = 'bullshit-image';
            textIndicator.className = 'bullshit-text-indicator';
            textIndicator.innerHTML = '🚫 <strong>BULLSHIT DETECTED!</strong>';
            textIndicator.title = 'This claim appears to be complete bullshit!';
            credibilityScoreContainer.appendChild(textIndicator);
            bullshitImage.remove();
            
            // Add animation effect for text indicator
            setTimeout(() => {
                textIndicator.classList.add('show');
            }, 100);
            return;
        };
        
        // Add the image to the credibility score container
        credibilityScoreContainer.appendChild(bullshitImage);
        
        // Add animation effect
        setTimeout(() => {
            bullshitImage.classList.add('show');
        }, 100);
    }
}

// Display Sentiment Analysis
function displaySentimentAnalysis(sentiment) {
    const sentimentContainer = document.getElementById('sentiment-analysis');
    
    let sentimentHTML = `
        <div class="sentiment-item sentiment-${sentiment.sentiment_category}">
            <div class="label">Sentiment</div>
            <div class="value">${sentiment.sentiment_category.charAt(0).toUpperCase() + sentiment.sentiment_category.slice(1)}</div>
        </div>
        <div class="sentiment-item">
            <div class="label">Polarity</div>
            <div class="value">${sentiment.polarity}</div>
        </div>
        <div class="sentiment-item">
            <div class="label">Subjectivity</div>
            <div class="value">${sentiment.subjectivity}</div>
        </div>
        <div class="sentiment-item">
            <div class="label">Objectivity</div>
            <div class="value">${sentiment.objectivity.charAt(0).toUpperCase() + sentiment.objectivity.slice(1)}</div>
        </div>
        <div class="sentiment-item">
            <div class="label">Confidence</div>
            <div class="value">${Math.round(sentiment.confidence * 100)}%</div>
        </div>
    `;
    
    // Add credibility level and reliability if available
    if (sentiment.credibility_level) {
        sentimentHTML += `
            <div class="sentiment-item">
                <div class="label">Credibility Level</div>
                <div class="value">${sentiment.credibility_level}</div>
            </div>
        `;
    }
    
    if (sentiment.reliability) {
        sentimentHTML += `
            <div class="sentiment-item">
                <div class="label">Reliability</div>
                <div class="value">${sentiment.reliability}</div>
            </div>
        `;
    }
    
    sentimentContainer.innerHTML = sentimentHTML;
}

// Display Recommendations
function displayRecommendations(recommendations) {
    const recommendationsContainer = document.getElementById('recommendations');
    
    if (!recommendations || recommendations.length === 0) {
        recommendationsContainer.innerHTML = '<p>No specific recommendations available.</p>';
        return;
    }
    
    const recommendationsHTML = recommendations.map(rec => 
        `<div class="recommendation-item">${rec}</div>`
    ).join('');
    
    recommendationsContainer.innerHTML = recommendationsHTML;
}

// Show Loading
function showLoading() {
    loading.classList.add('show');
    hideResults();
    
    // Disable buttons
    document.querySelectorAll('.verify-btn').forEach(btn => {
        btn.disabled = true;
    });
}

// Hide Loading
function hideLoading() {
    loading.classList.remove('show');
    
    // Enable buttons
    document.querySelectorAll('.verify-btn').forEach(btn => {
        btn.disabled = false;
    });
}

// Show Error
function showError(message) {
    hideLoading();
    
    // Create error notification
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-notification';
    errorDiv.innerHTML = `
        <div class="error-content">
            <i class="fas fa-exclamation-triangle"></i>
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    // Add error styles
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #fed7d7;
        color: #742a2a;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border: 2px solid #fc8181;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 1000;
        max-width: 400px;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(errorDiv);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (errorDiv.parentElement) {
            errorDiv.remove();
        }
    }, 5000);
}

// Hide Results
function hideResults() {
    resultsSection.classList.remove('show');
}

// Add CSS for error notification animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .error-content {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .error-content button {
        background: none;
        border: none;
        color: inherit;
        cursor: pointer;
        padding: 0.25rem;
        margin-left: auto;
    }
    
    .error-content button:hover {
        opacity: 0.7;
    }
`;
document.head.appendChild(style);

// Health Check
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('✅ API connection successful');
        } else {
            console.warn('⚠️ API health check failed');
        }
    } catch (error) {
        console.error('❌ API connection failed:', error);
        showError('Unable to connect to the verification service. Please ensure the backend is running.');
    }
}

// Check API health on page load
checkAPIHealth(); 