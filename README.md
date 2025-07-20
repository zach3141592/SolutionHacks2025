# Veridata 

Veridata is an AI-powered web application that verifies advertisement claims using Google's Gemini API and performs sentiment analysis to help users identify potentially misleading or false advertising statements.

## Features

- **AI-Powered Fact Checking**: Uses Google Gemini API to analyze and verify advertisement claims
- **Sentiment Analysis**: Provides detailed sentiment analysis including polarity, subjectivity, and objectivity
- **Text & File Input**: Support for both direct text input and file uploads
- **Credibility Scoring**: Assigns credibility scores from 0-100% based on AI analysis
- **Detailed Analysis**: Provides comprehensive breakdown of claims and potential issues
- **Recommendations**: Offers suggestions for improving transparency and accuracy
- **Modern UI**: Beautiful, responsive interface with smooth animations
- **Real-time Processing**: Fast verification with loading indicators and error handling

## Tech Stack

### Backend

- **FastAPI**: Modern, fast web framework for building APIs
- **Google Gemini AI**: Advanced language model for fact-checking and analysis
- **TextBlob**: Natural language processing library for sentiment analysis
- **Python Magic**: File type detection and content extraction
- **Pydantic**: Data validation and settings management

### Frontend

- **HTML5**: Semantic markup with modern web standards
- **CSS3**: Modern styling with animations and responsive design
- **JavaScript**: Interactive functionality and API integration
- **Font Awesome**: Beautiful icons and UI elements
- **Google Fonts**: Professional typography

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Modern web browser

### Setup Instructions

1. **Clone or download the project**

   ```bash
   cd /path/to/your/project/directory
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Create a `.env` file in the project root directory and add your Gemini API key:

   ```bash
   cp .env.example .env
   ```

   Then edit the `.env` file and replace `your_gemini_api_key_here` with your actual Google Gemini API key:

   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

   **To get a Gemini API key:**

   - Visit [Google AI Studio](https://aistudio.google.com/)
   - Sign in with your Google account
   - Go to "API Keys" section
   - Create a new API key
   - Copy the key to your `.env` file

4. **Set up the directory structure**
   ```
   Veridata/
   ├── backend/
   │   └── main.py
   ├── frontend/
   │   ├── index.html
   │   ├── styles.css
   │   └── script.js
   ├── .env                 # Your API key (not tracked in git)
   ├── .env.example         # Template for API key setup
   ├── .gitignore          # Excludes .env from version control
   ├── requirements.txt
   └── README.md
   ```

## Running the Application

### Start the Backend Server

1. Navigate to the project directory
2. Run the FastAPI server:

   ```bash
   python backend/main.py
   ```

   Or alternatively:

   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. The backend will be available at `http://localhost:8000`
4. API documentation will be available at `http://localhost:8000/docs`

### Access the Frontend

1. Open your web browser
2. Navigate to the `frontend` directory
3. Open `index.html` in your browser, or serve it using a local server:

   ```bash
   # Using Python's built-in server
   cd frontend
   python -m http.server 3000
   ```

   Then visit `http://localhost:3000`

## Usage

### Text Verification

1. Click on "Text Input" tab
2. Paste your advertisement text or claim into the textarea
3. Click "Verify Claim" button
4. Wait for the AI analysis to complete
5. Review the detailed results

### File Verification

1. Click on "File Upload" tab
2. Either drag and drop a file or click to browse
3. Select a text file containing advertisement content
4. Click "Verify File" button
5. Wait for processing and review results

### Understanding Results

**Verification Status:**

-  **Verified**: Claims appear to be factually accurate
-  **Partially Verified**: Some claims may need additional verification
-  **Unverified**: Claims lack sufficient supporting evidence
-  **Misleading**: Claims appear to be false or misleading

**Credibility Score:**

- **80-100%**: High credibility - Well-supported claims
- **60-79%**: Moderate credibility - Some verification needed
- **40-59%**: Low credibility - Several questionable claims
- **0-39%**: Very low credibility - Mostly misleading content

**Sentiment Analysis:**

- **Polarity**: Emotional tone (-1 to +1, negative to positive)
- **Subjectivity**: Opinion vs. fact (0 to 1, objective to subjective)
- **Objectivity**: Factual vs. opinion-based content
- **Confidence**: How certain the sentiment analysis is

## API Endpoints

### GET /

- Returns welcome message

### POST /verify-text

- Verifies text content
- Body: `{"text": "your advertisement text"}`
- Returns: Verification results with credibility score and analysis

### POST /verify-file

- Verifies uploaded file content
- Body: Form data with file upload
- Returns: Same verification results as text endpoint

### GET /health

- Returns API health status

##  Customization

### Modifying the Frontend

- Edit `frontend/styles.css` to change colors, fonts, or layout
- Modify `frontend/script.js` to add new features or change behavior
- Update `frontend/index.html` to add new sections or modify structure

### Backend Configuration

- Update `backend/main.py` to modify API endpoints or add new features
- Adjust the Gemini API prompt to change analysis focus
- Add new verification models or data sources

##  Security Notes

-  **API Key Security**: The Gemini API key is stored in a `.env` file that is excluded from version control
-  **Important**: Never commit your `.env` file to version control
-  **Production Setup**: For production deployment, use proper environment variable configuration
-  **Additional Security**: Consider adding rate limiting and user authentication for production use
-  **Input Validation**: All user inputs are validated and sanitized

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Troubleshooting

### Common Issues

**Backend won't start:**

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility (3.8+)
- Verify port 8000 is not in use by another application

**Frontend can't connect to backend:**

- Ensure backend is running on `http://localhost:8000`
- Check browser console for CORS errors
- Verify API endpoints are accessible

**Gemini API errors:**

- Check API key validity
- Verify internet connection
- Check API usage limits and quotas

**File upload issues:**

- Ensure files are text-based or readable
- Check file size limits
- Verify file permissions

### Getting Help

- Check the FastAPI documentation at `http://localhost:8000/docs`
- Review browser console for JavaScript errors
- Check backend logs for Python errors

## Features in Development

- Support for more file formats (PDF, Word documents)
- Batch processing for multiple advertisements
- Historical analysis and reporting
- User accounts and saved analyses
- API rate limiting and caching
- Multi-language support

---
