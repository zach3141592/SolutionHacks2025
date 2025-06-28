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
    """Extract text content from uploaded file"""
    try:
        # Read file content directly
        content = await file.read()
        
        # Check if file appears to be text-based by filename extension
        filename = file.filename or ""
        text_extensions = {'.txt', '.md', '.csv', '.json', '.xml', '.html', '.css', '.js', '.py', '.log'}
        is_likely_text = any(filename.lower().endswith(ext) for ext in text_extensions)
        
        # Try to decode as text with multiple encoding attempts
        text = None
        encodings_to_try = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                text = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        
        # If all encoding attempts failed, use utf-8 with error handling
        if text is None:
            text = content.decode('utf-8', errors='replace')
        
        # Basic validation - check if content looks like readable text
        if not text.strip():
            raise ValueError("No readable text content found in file")
            
        # Remove excessive control characters but keep basic formatting
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        if len(text.strip()) < 10:
            raise ValueError("File content too short or not readable as text")
            
        return text
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

async def verify_with_gemini(text: str) -> dict:
    """Use Gemini API to verify advertisement claims with fallback"""
    try:
        prompt = f"""
        You are an expert fact-checker and advertisement analyst. Please analyze the following advertisement text or claim:

        "{text}"

        Provide a comprehensive analysis including:
        1. Fact-checking: Verify if the claims made are factually accurate
        2. Credibility assessment: Rate the credibility from 0-100
        3. Potential misleading elements: Identify any potentially misleading statements
        4. Supporting evidence: What evidence would be needed to support these claims
        5. Recommendations: Suggest improvements for transparency and accuracy

        Respond in JSON format with the following structure:
        {{
            "verification_status": "verified/partially_verified/unverified/misleading",
            "credibility_score": 0-100,
            "fact_check_summary": "detailed analysis",
            "misleading_elements": ["list of potentially misleading elements"],
            "evidence_needed": ["list of evidence needed"],
            "recommendations": ["list of recommendations"]
        }}
        """

        response = model.generate_content(prompt)
        
        # Try to parse JSON response
        try:
            result = json.loads(response.text.strip())
        except json.JSONDecodeError:
            # If JSON parsing fails, create structured response from text
            result = {
                "verification_status": "analysis_completed",
                "credibility_score": 50,
                "fact_check_summary": response.text,
                "misleading_elements": [],
                "evidence_needed": [],
                "recommendations": []
            }
        
        return result
    except Exception as e:
        # Handle API quota exceeded and other errors gracefully
        print(f"Gemini API Error: {str(e)}")
        
        # Check if it's a quota exceeded error
        if "quota" in str(e).lower() or "429" in str(e):
            fallback_summary = "AI analysis temporarily unavailable due to API limits. Analysis based on local scoring algorithms only."
        else:
            fallback_summary = f"AI analysis unavailable: {str(e)[:100]}..."
        
        # Return fallback result - don't raise HTTPException
        return {
            "verification_status": "analysis_completed",
            "credibility_score": 60,  # Neutral fallback score
            "fact_check_summary": fallback_summary,
            "misleading_elements": [],
            "evidence_needed": ["External fact-checking recommended"],
            "recommendations": ["Verify claims through independent sources"]
        }

def analyze_sentiment(text: str) -> dict:
    """Perform sentiment analysis on the text"""
    try:
        blob = TextBlob(text)
        
        # Get polarity (-1 to 1) and subjectivity (0 to 1)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment category
        if polarity > 0.1:
            sentiment_category = "positive"
        elif polarity < -0.1:
            sentiment_category = "negative"  
        else:
            sentiment_category = "neutral"
            
        # Determine objectivity
        if subjectivity < 0.5:
            objectivity = "objective"
        else:
            objectivity = "subjective"
            
        return {
            "polarity": round(polarity, 3),
            "subjectivity": round(subjectivity, 3),
            "sentiment_category": sentiment_category,
            "objectivity": objectivity,
            "confidence": round(abs(polarity), 3)
        }
    except Exception as e:
        return {
            "polarity": 0,
            "subjectivity": 0,
            "sentiment_category": "neutral",
            "objectivity": "unknown",
            "confidence": 0,
            "error": str(e)
        }

def calculate_advanced_credibility_score(text: str, gemini_result: dict, sentiment_data: dict) -> dict:
    """Calculate a comprehensive credibility score based on multiple factors"""
    
    # Determine statement type for context-aware scoring
    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words)
    
    # Detect statement type
    is_health_fact = any(term in text_lower for term in [
        "health", "nutrition", "diet", "exercise", "sleep", "water", "food", "vitamin",
        "mineral", "protein", "fiber", "calcium", "antioxidant", "vegetable", "fruit"
    ])
    
    is_basic_fact = any(term in text_lower for term in [
        "is good", "is healthy", "is important", "is essential", "helps", "benefits",
        "prevents", "reduces", "supports", "aids", "provides"
    ]) and word_count <= 15
    
    is_scientific = any(term in text_lower for term in [
        "research", "study", "clinical", "peer-reviewed", "evidence", "data", "according to"
    ])
    
    # Initialize score components with context-aware base
    if is_basic_fact and is_health_fact:
        base_score = 75  # Higher for basic health facts
    elif is_scientific:
        base_score = 70  # Higher for scientific statements
    elif is_basic_fact:
        base_score = 68  # Higher for basic factual statements
    else:
        base_score = 60  # Standard for other statements
    
    scores = {
        "base_score": base_score,
        "language_analysis": 0,
        "sentiment_factor": 0,
        "red_flag_penalties": 0,
        "gemini_adjustment": 0,
        "factual_indicators": 0,
        "common_knowledge_bonus": 0,
        "statement_type_bonus": 0
    }
    
    # 1. Expanded common knowledge and factual statements
    common_facts = [
        # Core health facts (high confidence)
        ("vegetables are good", +20), ("vegetables are healthy", +20), ("fruits are healthy", +20),
        ("exercise is beneficial", +20), ("exercise is good", +20), ("exercise helps", +18),
        ("water is essential", +20), ("water is important", +18), ("drinking water", +15),
        ("sleep is important", +20), ("sleep is essential", +18), ("get enough sleep", +15),
        ("smoking is harmful", +20), ("smoking is bad", +18), ("don't smoke", +15),
        
        # Nutrition facts
        ("balanced diet", +18), ("healthy diet", +18), ("nutritious food", +15),
        ("eat vegetables", +15), ("eat fruits", +15), ("whole grains", +12),
        ("limit sugar", +15), ("reduce salt", +12), ("avoid processed", +12),
        
        # Basic science and health
        ("calcium is good for bones", +20), ("calcium strengthens bones", +18),
        ("fiber aids digestion", +20), ("fiber is good", +15), ("fiber helps", +15),
        ("protein builds muscle", +18), ("protein is important", +15),
        ("sun provides vitamin d", +20), ("vitamin d from sun", +18),
        ("vitamins are important", +15), ("minerals are essential", +15),
        ("antioxidants are beneficial", +15), ("omega-3", +12),
        
        # Exercise and fitness
        ("regular exercise", +18), ("physical activity", +15), ("cardio is good", +15),
        ("strength training", +12), ("stretching is important", +12),
        ("exercise reduces risk", +18), ("exercise prevents", +15),
        
        # General wellness
        ("maintain health", +15), ("prevent disease", +15), ("boost immunity", +12),
        ("support health", +12), ("promote wellness", +10), ("stay healthy", +12),
        ("healthy lifestyle", +15), ("wellness is important", +12)
    ]
    
    for phrase, bonus in common_facts:
        if phrase in text_lower:
            scores["common_knowledge_bonus"] += bonus
    
    # 2. Red flag detection (penalties) - only for suspicious claims
    red_flags = [
        # Extreme claims
        ("miracle cure", -25), ("instant cure", -20), ("doctors hate", -25), 
        ("one weird trick", -20), ("cure everything", -25), ("all diseases", -25),
        ("secret cure", -15), ("revolutionary breakthrough", -12),
        
        # Absolute false promises  
        ("100% effective", -15), ("never fails", -15), ("zero side effects", -12),
        ("completely safe", -8), ("guaranteed results", -10),
        
        # Pressure tactics
        ("limited time offer", -10), ("act now or", -10), ("today only", -10),
        ("while supplies last", -8), ("don't wait", -8),
        
        # Unrealistic promises
        ("lose 30 pounds overnight", -20), ("without any effort", -15),
        ("no diet no exercise", -15), ("instant weight loss", -18),
        
        # Conspiracy theories
        ("doctors don't want you to know", -20), ("big pharma conspiracy", -15),
        ("they don't want you to see", -18), ("hidden truth", -10),
        
        # Fantasy/Impossible claims
        ("magic", -30), ("magical", -25), ("supernatural", -25), ("mystical", -20),
        ("teleport", -25), ("time travel", -30), ("fly into the sky", -35), 
        ("levitate", -25), ("invisibility", -30), ("mind reading", -25),
        ("psychic powers", -25), ("crystal healing", -15), ("aura reading", -15),
        ("curse", -20), ("spell", -25), ("potion", -20), ("enchanted", -25),
        ("fairy tale", -30), ("unicorn", -35), ("dragon", -35), ("wizard", -30),
        ("take you into the sky", -35), ("fly without wings", -30), ("defy gravity", -25),
        ("travel through time", -35), ("read minds", -25), ("see the future", -25),
        ("turn invisible", -30), ("live forever", -30), ("never age", -25),
        ("breathe underwater without", -20), ("survive in space without", -25)
    ]
    
    for phrase, penalty in red_flags:
        if phrase in text_lower:
            scores["red_flag_penalties"] += penalty
    
    # 3. Language analysis - more nuanced
    # Only penalize excessive promotional language
    excessive_promo = ["amazing miracle", "incredible breakthrough", "unbelievable results",
                      "phenomenal discovery", "revolutionary secret", "extraordinary cure"]
    promo_count = sum(1 for phrase in excessive_promo if phrase in text_lower)
    
    # Single promotional words are less concerning
    single_promo = ["amazing", "incredible", "fantastic", "outstanding"]
    single_promo_count = sum(1 for word in single_promo if word in text_lower)
    
    scores["language_analysis"] = max(-15, -promo_count * 8 - single_promo_count * 2)
    
    # 4. Enhanced factual indicators
    scientific_terms = ["study", "research", "clinical trial", "peer-reviewed", 
                       "scientific evidence", "data shows", "university", "journal",
                       "fda approved", "clinical evidence", "peer reviewed"]
    factual_count = sum(1 for term in scientific_terms if term in text_lower)
    
    # Basic factual language
    basic_factual = ["may help", "can support", "studies suggest", "research shows",
                    "evidence indicates", "according to", "typically", "generally"]
    basic_count = sum(1 for term in basic_factual if term in text_lower)
    
    scores["factual_indicators"] = min(20, factual_count * 5 + basic_count * 3)
    
    # 5. Improved sentiment analysis
    if sentiment_data:
        subjectivity = sentiment_data.get("subjectivity", 0)
        polarity = sentiment_data.get("polarity", 0)
        
        # Be more lenient with health facts and basic statements
        if is_basic_fact or is_health_fact:
            # For basic facts, subjectivity is less concerning
            if subjectivity > 0.9 and any(word in text_lower for word in ["buy", "order", "purchase", "sale"]):
                subjectivity_penalty = -5
            elif subjectivity > 0.8:
                subjectivity_penalty = -2
            else:
                subjectivity_penalty = 0
        else:
            # Standard subjectivity handling for other statements
            if subjectivity > 0.8 and any(word in text_lower for word in ["buy", "order", "purchase", "sale"]):
                subjectivity_penalty = -8
            elif subjectivity > 0.6:
                subjectivity_penalty = -3
            else:
                subjectivity_penalty = 0
        
        # Improved polarity handling
        if polarity > 0.8 and not (is_basic_fact or is_health_fact):  # Only penalize extreme positivity in non-factual contexts
            polarity_penalty = -5
        elif polarity > 0.3:  # Moderately positive is good for health facts
            polarity_penalty = 3 if (is_basic_fact or is_health_fact) else 2
        elif polarity < -0.5:  # Very negative
            polarity_penalty = -3
        else:
            polarity_penalty = 0  # Neutral is fine
            
        scores["sentiment_factor"] = subjectivity_penalty + polarity_penalty
    
    # 6. Smart Gemini API adjustment - context aware
    gemini_score = gemini_result.get("credibility_score", 60)
    if isinstance(gemini_score, (int, float)):
        # Check if this is a fallback response (when Gemini API is unavailable)
        is_fallback = gemini_score == 60 and "unavailable" in gemini_result.get("fact_check_summary", "").lower()
        
        if is_fallback:
            # Don't penalize when Gemini is unavailable - rely on local algorithm
            gemini_adjustment = 0
        elif is_basic_fact and is_health_fact and gemini_score < 80:
            # Don't let Gemini pull down obvious health facts too much
            gemini_adjustment = max(-5, (gemini_score - 60) * 0.3)
        elif is_scientific and gemini_score >= 70:
            # Give more weight to Gemini for scientific claims when it's positive
            gemini_adjustment = (gemini_score - 60) * 0.8
        else:
            # Standard Gemini weighting
            gemini_adjustment = (gemini_score - 60) * 0.5
        
        scores["gemini_adjustment"] = gemini_adjustment
    
    # 7. Statement type bonus for context
    if is_basic_fact and is_health_fact and word_count <= 10:
        scores["statement_type_bonus"] = 5  # Bonus for simple health facts
    elif is_scientific:
        scores["statement_type_bonus"] = 3  # Bonus for scientific language
    
    # 8. Improved length analysis
    if word_count < 3:
        scores["language_analysis"] -= 8  # Very short might be incomplete
    elif word_count <= 10 and (is_basic_fact or is_health_fact):
        scores["language_analysis"] += 2  # Short factual statements are good
    elif word_count > 300:
        scores["language_analysis"] -= 5   # Very long might be misleading
    
    # Calculate final score
    final_score = (scores["base_score"] + 
                  scores["language_analysis"] + 
                  scores["sentiment_factor"] + 
                  scores["red_flag_penalties"] + 
                  scores["gemini_adjustment"] + 
                  scores["factual_indicators"] +
                  scores["common_knowledge_bonus"] +
                  scores["statement_type_bonus"])
    
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
            "base_score": scores["base_score"],
            "language_analysis": round(scores["language_analysis"], 1),
            "sentiment_factor": round(scores["sentiment_factor"], 1),
            "red_flag_penalties": round(scores["red_flag_penalties"], 1),
            "gemini_adjustment": round(scores["gemini_adjustment"], 1),
            "factual_indicators": round(scores["factual_indicators"], 1),
            "common_knowledge_bonus": round(scores["common_knowledge_bonus"], 1),
            "statement_type_bonus": round(scores["statement_type_bonus"], 1)
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
        
        # Calculate advanced credibility score
        credibility_analysis = calculate_advanced_credibility_score(
            request.text, gemini_result, sentiment_result
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
            fact_check_details=gemini_result.get("fact_check_summary", "No analysis available"),
            recommendations=gemini_result.get("recommendations", [])
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
        
        # Calculate advanced credibility score
        credibility_analysis = calculate_advanced_credibility_score(
            text, gemini_result, sentiment_result
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
            fact_check_details=gemini_result.get("fact_check_summary", "No analysis available"),
            recommendations=gemini_result.get("recommendations", [])
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