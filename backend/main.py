from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from textblob import TextBlob
import aiofiles
import os
from typing import Optional
import json
from dotenv import load_dotenv
import io
import tempfile
from pathlib import Path

# Computer Vision and OCR imports
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_bytes, convert_from_path
import PyPDF2
import base64

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Veridata", description="Advertisement Verification Service")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please create a .env file with your API key.")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

class TextRequest(BaseModel):
    text: str

class VerificationResult(BaseModel):
    original_text: str
    verification_result: str
    credibility_score: float
    sentiment_analysis: dict
    fact_check_details: str
    recommendations: list

async def extract_text_from_file(file: UploadFile) -> str:
    """Advanced text extraction from various file types using computer vision and OCR"""
    try:
        # Read file content
        content = await file.read()
        filename = file.filename or ""
        file_extension = Path(filename).suffix.lower()
        
        print(f"Processing file: {filename} ({file_extension})")
        
        # Determine file type and processing method
        if file_extension in {'.txt', '.md', '.csv', '.json', '.xml', '.html', '.css', '.js', '.py', '.log'}:
            return await process_text_file(content)
        elif file_extension in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}:
            return await process_image_file(content, filename)
        elif file_extension == '.pdf':
            return await process_pdf_file(content, filename)
        else:
            # Try as text first, then as image
            try:
                return await process_text_file(content)
            except:
                try:
                    return await process_image_file(content, filename)
                except:
                    raise ValueError(f"Unsupported file type: {file_extension}")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

async def process_text_file(content: bytes) -> str:
    """Process text-based files with encoding detection"""
    encodings_to_try = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'ascii']
    
    for encoding in encodings_to_try:
        try:
            text = content.decode(encoding)
            if text.strip():
                # Clean text by removing excessive control characters
                text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
                
                if len(text.strip()) >= 10:
                    return text
        except UnicodeDecodeError:
            continue
    
    # Final attempt with error handling
    text = content.decode('utf-8', errors='replace')
    if len(text.strip()) < 10:
        raise ValueError("File content too short or not readable as text")
    return text

async def process_image_file(content: bytes, filename: str) -> str:
    """Process image files using advanced OCR techniques"""
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(content))
        
        # Convert to RGB if necessary
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        print(f"Image loaded: {image.size}, mode: {image.mode}")
        
        # Apply image preprocessing for better OCR
        processed_image = preprocess_image_for_ocr(image)
        
        # Perform OCR with multiple configurations
        extracted_text = perform_advanced_ocr(processed_image)
        
        if not extracted_text or len(extracted_text.strip()) < 5:
            # Try with original image if preprocessing didn't help
            extracted_text = perform_advanced_ocr(image)
        
        if not extracted_text or len(extracted_text.strip()) < 3:
            raise ValueError("No readable text found in image")
        
        print(f"Extracted text length: {len(extracted_text)}")
        print(f"Extracted text content: '{extracted_text.strip()}'")
        return extracted_text.strip()
        
    except Exception as e:
        raise ValueError(f"Failed to extract text from image: {str(e)}")

def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """Apply image preprocessing techniques to improve OCR accuracy"""
    try:
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to PIL Image
        processed_image = Image.fromarray(cleaned)
        
        # Additional PIL-based enhancements
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(processed_image)
        processed_image = enhancer.enhance(2.0)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(processed_image)
        processed_image = enhancer.enhance(2.0)
        
        return processed_image
        
    except Exception as e:
        print(f"Preprocessing failed, using original image: {e}")
        return image

def perform_advanced_ocr(image: Image.Image) -> str:
    """Perform OCR with multiple configurations for best results"""
    
    # OCR configurations to try
    ocr_configs = [
        '--oem 3 --psm 6',  # Default: Assume uniform block of text
        '--oem 3 --psm 4',  # Assume single column of text
        '--oem 3 --psm 3',  # Fully automatic page segmentation
        '--oem 3 --psm 1',  # Automatic page segmentation with OSD
        '--oem 3 --psm 11', # Sparse text
        '--oem 3 --psm 12', # Sparse text with OSD
        '--oem 3 --psm 8',  # Single word
        '--oem 3 --psm 7',  # Single text line
    ]
    
    best_text = ""
    best_confidence = 0
    
    for config in ocr_configs:
        try:
            # Get text with confidence data
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Extract text
            text = pytesseract.image_to_string(image, config=config).strip()
            
            # Keep the result with highest confidence and reasonable length
            if avg_confidence > best_confidence and len(text) > 10:
                best_text = text
                best_confidence = avg_confidence
                
        except Exception as e:
            print(f"OCR config {config} failed: {e}")
            continue
    
    # If no good result, try basic OCR
    if not best_text:
        try:
            best_text = pytesseract.image_to_string(image).strip()
        except Exception as e:
            print(f"Basic OCR failed: {e}")
    
    return best_text

async def process_pdf_file(content: bytes, filename: str) -> str:
    """Process PDF files with text extraction and OCR fallback"""
    try:
        # First try direct text extraction
        text = extract_text_from_pdf(content)
        
        if text and len(text.strip()) >= 50:
            print("PDF text extracted directly")
            return text
        
        # If direct extraction fails or yields little text, use OCR
        print("PDF direct extraction failed, trying OCR...")
        text = extract_text_from_pdf_with_ocr(content)
        
        if not text or len(text.strip()) < 10:
            raise ValueError("No readable text found in PDF")
        
        return text
        
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

def extract_text_from_pdf(content: bytes) -> str:
    """Extract text directly from PDF using PyPDF2"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        text = ""
        
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        return text.strip()
        
    except Exception as e:
        print(f"Direct PDF extraction failed: {e}")
        return ""

def extract_text_from_pdf_with_ocr(content: bytes) -> str:
    """Extract text from PDF using OCR (for scanned PDFs)"""
    try:
        # Convert PDF pages to images
        images = convert_from_bytes(content, dpi=300, fmt='PNG')
        
        extracted_text = ""
        
        for i, image in enumerate(images):
            print(f"Processing PDF page {i+1}/{len(images)}")
            
            # Preprocess image for better OCR
            processed_image = preprocess_image_for_ocr(image)
            
            # Perform OCR
            page_text = perform_advanced_ocr(processed_image)
            
            if page_text:
                extracted_text += f"\n--- Page {i+1} ---\n{page_text}\n"
        
        return extracted_text.strip()
        
    except Exception as e:
        print(f"PDF OCR extraction failed: {e}")
        raise e

async def verify_with_gemini(text: str) -> dict:
    """Use Gemini API to verify advertisement claims and get accuracy score"""
    try:
        print(f"Analyzing text with Gemini: '{text}' (length: {len(text)})")
        
        # Handle very short text
        if len(text.strip()) < 5:
            return {
                "accuracy_score": 50,
                "reasoning": "Text too short for meaningful analysis",
                "main_issues": ["Insufficient text content"],
                "verification_status": "insufficient_data"
            }
        
        prompt = f"""
        You are an expert fact-checker analyzing advertisement claims. Analyze this text:

        "{text}"

        Rate the ACCURACY and TRUTHFULNESS of these claims on a scale of 0-100, where:
        - 90-100: Highly accurate, well-supported factual claims
        - 70-89: Mostly accurate with minor issues or exaggerations
        - 50-69: Partially accurate but with significant concerns
        - 30-49: Mostly inaccurate or misleading
        - 0-29: Completely false, impossible, or fraudulent claims

        Consider:
        - Scientific accuracy and plausibility
        - Whether claims can be substantiated
        - Presence of impossible or fantasy elements
        - Misleading or exaggerated language
        - Overall truthfulness

        Respond in JSON format:
        {{
            "accuracy_score": [0-100 number],
            "reasoning": "brief explanation of the score",
            "main_issues": ["list of key problems if any"],
            "verification_status": "accurate/concerning/misleading/false"
        }}
        """

        response = model.generate_content(prompt)
        
        # Try to parse JSON response (handle markdown formatting)
        try:
            response_text = response.text.strip()
            print(f"Raw Gemini response: '{response_text}'")
            
            # Remove markdown code block formatting if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove closing ```
            
            print(f"Cleaned response for JSON parsing: '{response_text.strip()}'")
            result = json.loads(response_text.strip())
            print(f"Parsed JSON result: {result}")
            
            # Ensure we have the expected fields
            if "accuracy_score" not in result:
                result["accuracy_score"] = 50
            return result
        except json.JSONDecodeError:
            # If JSON parsing fails, create structured response from text
            return {
                "accuracy_score": 50,
                "reasoning": response.text[:200] + "..." if len(response.text) > 200 else response.text,
                "main_issues": [],
                "verification_status": "analysis_completed"
            }
        
    except Exception as e:
        # Handle API quota exceeded and other errors gracefully
        print(f"Gemini API Error: {str(e)}")
        
        # Check if it's a quota exceeded error
        if "quota" in str(e).lower() or "429" in str(e):
            reasoning = "AI analysis temporarily unavailable due to API limits."
        else:
            reasoning = f"AI analysis unavailable: {str(e)[:100]}..."
        
        # Return fallback result - don't raise HTTPException
        return {
            "accuracy_score": 50,  # Neutral fallback score
            "reasoning": reasoning,
            "main_issues": ["AI analysis unavailable"],
            "verification_status": "analysis_unavailable"
        }

def analyze_sentiment(text: str) -> dict:
    """Perform advanced sentiment analysis tailored for advertisement verification"""
    try:
        blob = TextBlob(text)
        text_lower = text.lower()
        words = text_lower.split()
        
        # Basic TextBlob analysis
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Advanced emotional analysis
        emotional_analysis = analyze_emotional_patterns(text_lower, words)
        
        # Advertisement manipulation detection
        manipulation_analysis = detect_manipulation_tactics(text_lower)
        
        # Language pressure analysis
        pressure_analysis = analyze_pressure_language(text_lower)
        
        # Determine enhanced sentiment category
        sentiment_category = determine_sentiment_category(polarity, emotional_analysis)
        
        # Determine objectivity with more nuance
        objectivity_analysis = analyze_objectivity(subjectivity, text_lower, words)
        
        # Calculate overall sentiment trustworthiness
        trustworthiness = calculate_sentiment_trustworthiness(
            polarity, subjectivity, manipulation_analysis, pressure_analysis
        )
        
        # Determine confidence level
        confidence = calculate_sentiment_confidence(polarity, subjectivity, len(words))
        
        return {
            "polarity": round(polarity, 3),
            "subjectivity": round(subjectivity, 3),
            "sentiment_category": sentiment_category,
            "objectivity": objectivity_analysis["category"],
            "confidence": round(confidence, 3),
            "trustworthiness_score": round(trustworthiness, 1),
            "emotional_intensity": emotional_analysis["intensity"],
            "emotional_indicators": emotional_analysis["indicators"],
            "manipulation_detected": manipulation_analysis["detected"],
            "manipulation_tactics": manipulation_analysis["tactics"],
            "pressure_level": pressure_analysis["level"],
            "pressure_indicators": pressure_analysis["indicators"],
            "objectivity_details": objectivity_analysis,
            "language_analysis": {
                "word_count": len(words),
                "exclamation_count": text.count("!"),
                "question_count": text.count("?"),
                "caps_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0
            }
        }
    except Exception as e:
        return {
            "polarity": 0,
            "subjectivity": 0,
            "sentiment_category": "neutral",
            "objectivity": "unknown",
            "confidence": 0,
            "trustworthiness_score": 50,
            "emotional_intensity": "low",
            "error": str(e)
        }

def analyze_emotional_patterns(text_lower: str, words: list) -> dict:
    """Analyze emotional patterns and intensity in the text"""
    
    # Emotional intensity indicators
    high_intensity_words = [
        "amazing", "incredible", "unbelievable", "fantastic", "extraordinary", "phenomenal",
        "revolutionary", "breakthrough", "miracle", "ultimate", "perfect", "guaranteed",
        "instant", "immediate", "dramatic", "shocking", "stunning", "overwhelming"
    ]
    
    moderate_intensity_words = [
        "great", "excellent", "wonderful", "awesome", "impressive", "remarkable",
        "effective", "powerful", "strong", "significant", "important", "valuable"
    ]
    
    emotional_words = [
        "love", "hate", "fear", "excited", "thrilled", "devastated", "overjoyed",
        "furious", "delighted", "horrified", "ecstatic", "terrified", "passionate"
    ]
    
    # Count emotional indicators
    high_count = sum(1 for word in high_intensity_words if word in text_lower)
    moderate_count = sum(1 for word in moderate_intensity_words if word in text_lower)
    emotional_count = sum(1 for word in emotional_words if word in text_lower)
    
    # Calculate intensity
    total_words = len(words)
    if total_words == 0:
        intensity_ratio = 0
    else:
        intensity_ratio = (high_count * 3 + moderate_count * 2 + emotional_count) / total_words
    
    # Determine intensity level
    if intensity_ratio > 0.15:
        intensity = "very_high"
    elif intensity_ratio > 0.08:
        intensity = "high"
    elif intensity_ratio > 0.04:
        intensity = "moderate"
    elif intensity_ratio > 0.01:
        intensity = "low"
    else:
        intensity = "minimal"
    
    # Identify specific emotional indicators
    indicators = []
    if high_count > 0:
        indicators.append(f"High-intensity words: {high_count}")
    if moderate_count > 0:
        indicators.append(f"Moderate-intensity words: {moderate_count}")
    if emotional_count > 0:
        indicators.append(f"Emotional words: {emotional_count}")
    
    return {
        "intensity": intensity,
        "intensity_ratio": round(intensity_ratio, 4),
        "indicators": indicators,
        "high_intensity_count": high_count,
        "moderate_intensity_count": moderate_count,
        "emotional_count": emotional_count
    }

def detect_manipulation_tactics(text_lower: str) -> dict:
    """Detect common manipulation tactics in advertising"""
    
    manipulation_patterns = {
        "scarcity": [
            "limited time", "while supplies last", "only today", "act now", "hurry",
            "don't miss out", "last chance", "limited offer", "expires soon", "few left"
        ],
        "social_proof": [
            "everyone is buying", "most popular", "bestseller", "thousands sold",
            "customer favorite", "highly rated", "top choice", "preferred by"
        ],
        "authority": [
            "doctors recommend", "experts say", "scientists prove", "studies show",
            "endorsed by", "approved by", "recommended by", "trusted by"
        ],
        "fear_mongering": [
            "don't let", "avoid disaster", "protect yourself", "before it's too late",
            "dangerous", "risky", "harmful", "devastating consequences"
        ],
        "false_urgency": [
            "act now", "don't wait", "immediate action", "urgent", "time sensitive",
            "deadline approaching", "offer ends", "final hours"
        ],
        "exaggeration": [
            "completely", "totally", "absolutely", "100%", "never", "always",
            "everyone", "nobody", "all", "none", "perfect", "flawless"
        ]
    }
    
    detected_tactics = []
    tactic_details = {}
    
    for tactic, phrases in manipulation_patterns.items():
        matches = [phrase for phrase in phrases if phrase in text_lower]
        if matches:
            detected_tactics.append(tactic)
            tactic_details[tactic] = matches
    
    manipulation_score = len(detected_tactics) / len(manipulation_patterns) * 100
    
    return {
        "detected": len(detected_tactics) > 0,
        "tactics": detected_tactics,
        "details": tactic_details,
        "manipulation_score": round(manipulation_score, 1),
        "severity": "high" if len(detected_tactics) >= 3 else "moderate" if len(detected_tactics) >= 2 else "low" if len(detected_tactics) >= 1 else "none"
    }

def analyze_pressure_language(text_lower: str) -> dict:
    """Analyze language that creates psychological pressure"""
    
    pressure_indicators = {
        "high_pressure": [
            "act now or", "don't miss", "last chance", "final warning", "urgent",
            "immediate action required", "time running out", "deadline"
        ],
        "moderate_pressure": [
            "limited time", "special offer", "today only", "while supplies last",
            "don't wait", "hurry", "fast action"
        ],
        "low_pressure": [
            "consider", "think about", "when you're ready", "at your convenience",
            "take your time", "no rush", "whenever"
        ]
    }
    
    high_pressure_count = sum(1 for phrase in pressure_indicators["high_pressure"] if phrase in text_lower)
    moderate_pressure_count = sum(1 for phrase in pressure_indicators["moderate_pressure"] if phrase in text_lower)
    low_pressure_count = sum(1 for phrase in pressure_indicators["low_pressure"] if phrase in text_lower)
    
    # Determine pressure level
    if high_pressure_count >= 2:
        level = "very_high"
    elif high_pressure_count >= 1 or moderate_pressure_count >= 3:
        level = "high"
    elif moderate_pressure_count >= 1:
        level = "moderate"
    elif low_pressure_count >= 1:
        level = "low"
    else:
        level = "neutral"
    
    indicators = []
    if high_pressure_count > 0:
        indicators.append(f"High-pressure phrases: {high_pressure_count}")
    if moderate_pressure_count > 0:
        indicators.append(f"Moderate-pressure phrases: {moderate_pressure_count}")
    if low_pressure_count > 0:
        indicators.append(f"Low-pressure phrases: {low_pressure_count}")
    
    return {
        "level": level,
        "indicators": indicators,
        "high_pressure_count": high_pressure_count,
        "moderate_pressure_count": moderate_pressure_count,
        "low_pressure_count": low_pressure_count
    }

def determine_sentiment_category(polarity: float, emotional_analysis: dict) -> str:
    """Determine sentiment category with enhanced granularity"""
    
    intensity = emotional_analysis["intensity"]
    
    if polarity > 0.6:
        return "very_positive" if intensity in ["high", "very_high"] else "positive"
    elif polarity > 0.2:
        return "moderately_positive"
    elif polarity > -0.2:
        return "neutral"
    elif polarity > -0.6:
        return "moderately_negative"
    else:
        return "very_negative" if intensity in ["high", "very_high"] else "negative"

def analyze_objectivity(subjectivity: float, text_lower: str, words: list) -> dict:
    """Analyze objectivity with more nuanced categorization"""
    
    # Objective indicators
    objective_indicators = [
        "according to", "research shows", "studies indicate", "data suggests",
        "statistics show", "evidence indicates", "fact", "proven", "measured"
    ]
    
    # Subjective indicators
    subjective_indicators = [
        "i think", "i believe", "in my opinion", "i feel", "personally",
        "amazing", "terrible", "wonderful", "awful", "love", "hate"
    ]
    
    objective_count = sum(1 for phrase in objective_indicators if phrase in text_lower)
    subjective_count = sum(1 for phrase in subjective_indicators if phrase in text_lower)
    
    # Enhanced objectivity categorization
    if subjectivity < 0.2:
        category = "highly_objective"
    elif subjectivity < 0.4:
        category = "mostly_objective"
    elif subjectivity < 0.6:
        category = "balanced"
    elif subjectivity < 0.8:
        category = "mostly_subjective"
    else:
        category = "highly_subjective"
    
    return {
        "category": category,
        "subjectivity_score": round(subjectivity, 3),
        "objective_indicators": objective_count,
        "subjective_indicators": subjective_count,
        "objectivity_ratio": round(1 - subjectivity, 3)
    }

def calculate_sentiment_trustworthiness(polarity: float, subjectivity: float, 
                                      manipulation_analysis: dict, pressure_analysis: dict) -> float:
    """Calculate overall trustworthiness based on sentiment factors"""
    
    base_score = 50  # Neutral starting point
    
    # Polarity adjustment (extreme emotions are suspicious in ads)
    if abs(polarity) > 0.8:
        base_score -= 15  # Very extreme sentiment is suspicious
    elif abs(polarity) > 0.5:
        base_score -= 5   # Moderate extreme sentiment
    elif 0.1 <= abs(polarity) <= 0.3:
        base_score += 10  # Mild sentiment is more trustworthy
    
    # Subjectivity adjustment (objective is more trustworthy)
    objectivity_bonus = (1 - subjectivity) * 20
    base_score += objectivity_bonus
    
    # Manipulation penalty
    manipulation_penalty = manipulation_analysis["manipulation_score"] * 0.3
    base_score -= manipulation_penalty
    
    # Pressure penalty
    pressure_penalties = {
        "very_high": -20,
        "high": -15,
        "moderate": -8,
        "low": 5,
        "neutral": 0
    }
    base_score += pressure_penalties.get(pressure_analysis["level"], 0)
    
    return max(0, min(100, base_score))

def calculate_sentiment_confidence(polarity: float, subjectivity: float, word_count: int) -> float:
    """Calculate confidence in sentiment analysis"""
    
    # Base confidence from polarity strength
    polarity_confidence = abs(polarity)
    
    # Text length factor (more text = more confidence)
    length_factor = min(1.0, word_count / 20)  # Normalize to 20 words
    
    # Subjectivity factor (clear subjectivity = more confidence)
    subjectivity_confidence = abs(subjectivity - 0.5) * 2  # Distance from neutral
    
    # Combined confidence
    confidence = (polarity_confidence * 0.5 + 
                 length_factor * 0.3 + 
                 subjectivity_confidence * 0.2)
    
    return min(1.0, confidence)

def calculate_credibility_score(gemini_result: dict, sentiment_data: dict) -> dict:
    """Calculate credibility score: 90% Gemini AI + 10% sentiment analysis"""
    
    # Get Gemini accuracy score (0-100)
    gemini_score = gemini_result.get("accuracy_score", 50)
    
    # Use advanced sentiment analysis for credibility scoring
    sentiment_score = 50  # Default neutral
    if sentiment_data:
        # Use the comprehensive trustworthiness score from advanced sentiment analysis
        base_trustworthiness = sentiment_data.get("trustworthiness_score", 50)
        
        # Additional factors from advanced analysis
        manipulation_detected = sentiment_data.get("manipulation_detected", False)
        pressure_level = sentiment_data.get("pressure_level", "neutral")
        emotional_intensity = sentiment_data.get("emotional_intensity", "minimal")
        
        # Start with base trustworthiness
        sentiment_score = base_trustworthiness
        
        # Apply additional penalties for suspicious patterns
        if manipulation_detected:
            manipulation_tactics = sentiment_data.get("manipulation_tactics", [])
            manipulation_penalty = len(manipulation_tactics) * 5  # 5 points per tactic
            sentiment_score -= manipulation_penalty
        
        # Pressure level adjustments
        pressure_adjustments = {
            "very_high": -15,
            "high": -10,
            "moderate": -5,
            "low": 0,
            "neutral": 0
        }
        sentiment_score += pressure_adjustments.get(pressure_level, 0)
        
        # Emotional intensity adjustments (very high intensity in ads is suspicious)
        intensity_adjustments = {
            "very_high": -10,
            "high": -5,
            "moderate": 0,
            "low": 5,
            "minimal": 0
        }
        sentiment_score += intensity_adjustments.get(emotional_intensity, 0)
        
        # Ensure score stays within bounds
        sentiment_score = max(0, min(100, sentiment_score))
    
    # Dynamic weighting based on Gemini's confidence
    # When Gemini detects serious issues (very low scores), reduce sentiment influence
    if gemini_score <= 20:  # Extremely low Gemini score = trust Gemini more
        gemini_weight = 0.95
        sentiment_weight = 0.05
    elif gemini_score <= 40:  # Low Gemini score = still trust Gemini more
        gemini_weight = 0.93
        sentiment_weight = 0.07
    else:  # Normal scores = standard weighting
        gemini_weight = 0.90
        sentiment_weight = 0.10
    
    final_score = (gemini_score * gemini_weight) + (sentiment_score * sentiment_weight)
    
    # Ensure score is between 0 and 100
    final_score = max(0, min(100, final_score))
    
    # Improved credibility level determination
    if final_score >= 90:
        credibility_level = "Exceptional"
        reliability = "Highly Trustworthy"
    elif final_score >= 80:
        credibility_level = "Very High"
        reliability = "Very Trustworthy"
    elif final_score >= 70:
        credibility_level = "High"
        reliability = "Trustworthy"
    elif final_score >= 60:
        credibility_level = "Moderate"
        reliability = "Somewhat Trustworthy"
    elif final_score >= 45:
        credibility_level = "Low"
        reliability = "Questionable"
    elif final_score >= 30:
        credibility_level = "Very Low"
        reliability = "Likely Misleading"
    else:
        credibility_level = "Extremely Low"
        reliability = "Highly Misleading"
    
    return {
        "final_score": round(final_score, 1),
        "credibility_level": credibility_level,
        "reliability": reliability,
        "score_breakdown": {
            "gemini_score": round(gemini_score, 1),
            "gemini_weight": f"{gemini_weight*100:.0f}%",
            "sentiment_score": round(sentiment_score, 1),
            "sentiment_weight": f"{sentiment_weight*100:.0f}%",
            "reasoning": gemini_result.get("reasoning", "No reasoning provided"),
            "sentiment_details": {
                "trustworthiness": sentiment_data.get("trustworthiness_score", 50) if sentiment_data else 50,
                "manipulation_detected": sentiment_data.get("manipulation_detected", False) if sentiment_data else False,
                "pressure_level": sentiment_data.get("pressure_level", "neutral") if sentiment_data else "neutral",
                "emotional_intensity": sentiment_data.get("emotional_intensity", "minimal") if sentiment_data else "minimal",
                "objectivity": sentiment_data.get("objectivity", "unknown") if sentiment_data else "unknown"
            }
        }
    }

@app.get("/")
async def root():
    return {"message": "Welcome to Veridata - Advertisement Verification Service"}

@app.post("/verify-text", response_model=VerificationResult)
async def verify_text(request: TextRequest):
    """Verify advertisement text for accuracy and sentiment"""
    try:
        # Get Gemini verification
        gemini_result = await verify_with_gemini(request.text)
        
        # Perform sentiment analysis
        sentiment_result = analyze_sentiment(request.text)
        
        # Calculate credibility score
        credibility_analysis = calculate_credibility_score(
            gemini_result, sentiment_result
        )
        
        # Enhanced sentiment analysis with credibility factors
        enhanced_sentiment = sentiment_result.copy()
        enhanced_sentiment["credibility_level"] = credibility_analysis["credibility_level"]
        enhanced_sentiment["reliability"] = credibility_analysis["reliability"]
        enhanced_sentiment["score_breakdown"] = credibility_analysis["score_breakdown"]
        
        # Compile results
        result = VerificationResult(
            original_text=request.text,
            verification_result=gemini_result.get("verification_status", "unknown"),
            credibility_score=credibility_analysis["final_score"] / 100.0,
            sentiment_analysis=enhanced_sentiment,
            fact_check_details=gemini_result.get("reasoning", "No analysis available"),
            recommendations=gemini_result.get("main_issues", [])
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/verify-file", response_model=VerificationResult)
async def verify_file(file: UploadFile = File(...)):
    """Verify advertisement file for accuracy and sentiment"""
    try:
        # Extract text from file
        text = await extract_text_from_file(file)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text content found in file")
        
        # Get Gemini verification
        gemini_result = await verify_with_gemini(text)
        
        # Perform sentiment analysis
        sentiment_result = analyze_sentiment(text)
        
        # Calculate credibility score
        credibility_analysis = calculate_credibility_score(
            gemini_result, sentiment_result
        )
        
        # Enhanced sentiment analysis with credibility factors
        enhanced_sentiment = sentiment_result.copy()
        enhanced_sentiment["credibility_level"] = credibility_analysis["credibility_level"]
        enhanced_sentiment["reliability"] = credibility_analysis["reliability"]
        enhanced_sentiment["score_breakdown"] = credibility_analysis["score_breakdown"]
        
        # Compile results
        result = VerificationResult(
            original_text=text[:500] + "..." if len(text) > 500 else text,
            verification_result=gemini_result.get("verification_status", "unknown"),
            credibility_score=credibility_analysis["final_score"] / 100.0,
            sentiment_analysis=enhanced_sentiment,
            fact_check_details=gemini_result.get("reasoning", "No analysis available"),
            recommendations=gemini_result.get("main_issues", [])
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Veridata API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 