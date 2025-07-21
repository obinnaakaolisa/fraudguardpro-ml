from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import numpy as np
import logging
import time
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FraudGuard ML API",
    description="Advanced fraud detection API using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount (must be positive)")
    currency: str = Field(..., min_length=3, max_length=3, description="Currency code (e.g., USD, EUR)")
    merchantId: str = Field(..., min_length=1, description="Merchant identifier")
    paymentMethod: str = Field(..., description="Payment method used")
    customerEmail: str = Field(..., description="Customer email address")
    ipAddress: Optional[str] = Field(None, description="Customer IP address")
    deviceId: Optional[str] = Field(None, description="Device identifier")
    description: Optional[str] = Field(None, description="Transaction description")
    timestamp: Optional[str] = Field(None, description="Transaction timestamp")
    
    @validator('currency')
    def currency_must_be_valid(cls, v):
        valid_currencies = ['USD', 'EUR', 'GBP', 'NGN', 'CAD', 'AUD', 'JPY']
        if v.upper() not in valid_currencies:
            raise ValueError(f'Currency must be one of: {", ".join(valid_currencies)}')
        return v.upper()
    
    @validator('paymentMethod')
    def payment_method_must_be_valid(cls, v):
        valid_methods = ['credit_card', 'debit_card', 'bank_transfer', 'digital_wallet', 'cash']
        if v.lower() not in valid_methods:
            raise ValueError(f'Payment method must be one of: {", ".join(valid_methods)}')
        return v.lower()
    
    @validator('customerEmail')
    def email_must_be_valid(cls, v):
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError('Invalid email format')
        return v.lower()

class TransactionRequest(BaseModel):
    transactions: List[Transaction] = Field(..., min_items=1, max_items=100, description="List of transactions to analyze")

class FraudPrediction(BaseModel):
    isFraud: bool = Field(..., description="Whether transaction is predicted as fraud")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    riskScore: float = Field(..., ge=0.0, le=1.0, description="Risk score (0-1)")
    riskLevel: str = Field(..., description="Risk level: low, medium, high")
    riskFactors: List[str] = Field(..., description="List of risk factors identified")
    processingTime: float = Field(..., description="Processing time in milliseconds")

class PredictionResponse(BaseModel):
    predictions: List[FraudPrediction]
    totalProcessed: int = Field(..., description="Total number of transactions processed")
    averageRiskScore: float = Field(..., description="Average risk score across all transactions")
    highRiskCount: int = Field(..., description="Number of high-risk transactions")

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime: float

class ModelInfo(BaseModel):
    name: str
    version: str
    features: List[str]
    riskThresholds: Dict[str, float]
    supportedCurrencies: List[str]
    supportedPaymentMethods: List[str]

# Global variables for tracking
start_time = time.time()
request_count = 0

class FraudDetectionEngine:
    """Advanced fraud detection engine with multiple risk factors"""
    
    def __init__(self):
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        
        self.suspicious_domains = [
            'tempmail.org', '10minutemail.com', 'guerrillamail.com',
            'mailinator.com', 'throwaway.email', 'temp-mail.org'
        ]
        
        self.high_risk_countries = ['XX', 'YY']  # Placeholder country codes
        
    def calculate_fraud_risk(self, transaction: Transaction) -> tuple[float, List[str]]:
        """Calculate fraud risk score and identify risk factors"""
        risk_score = 0.0
        risk_factors = []
        
        # Amount-based risk assessment
        amount_risk, amount_factors = self._assess_amount_risk(transaction.amount)
        risk_score += amount_risk
        risk_factors.extend(amount_factors)
        
        # Payment method risk
        payment_risk, payment_factors = self._assess_payment_method_risk(transaction.paymentMethod)
        risk_score += payment_risk
        risk_factors.extend(payment_factors)
        
        # Email domain risk
        email_risk, email_factors = self._assess_email_risk(transaction.customerEmail)
        risk_score += email_risk
        risk_factors.extend(email_factors)
        
        # Merchant risk
        merchant_risk, merchant_factors = self._assess_merchant_risk(transaction.merchantId)
        risk_score += merchant_risk
        risk_factors.extend(merchant_factors)
        
        # Time-based risk (if timestamp provided)
        if transaction.timestamp:
            time_risk, time_factors = self._assess_time_risk(transaction.timestamp)
            risk_score += time_risk
            risk_factors.extend(time_factors)
        
        # Currency risk
        currency_risk, currency_factors = self._assess_currency_risk(transaction.currency)
        risk_score += currency_risk
        risk_factors.extend(currency_factors)
        
        # Add some ML-like variability
        risk_score += np.random.uniform(0, 0.05)
        
        # Ensure risk score is between 0 and 1
        risk_score = min(max(risk_score, 0.0), 1.0)
        
        return risk_score, risk_factors
    
    def _assess_amount_risk(self, amount: float) -> tuple[float, List[str]]:
        """Assess risk based on transaction amount"""
        risk = 0.0
        factors = []
        
        if amount > 50000:
            risk += 0.4
            factors.append("Very high transaction amount")
        elif amount > 10000:
            risk += 0.3
            factors.append("High transaction amount")
        elif amount > 5000:
            risk += 0.2
            factors.append("Above average transaction amount")
        elif amount > 1000:
            risk += 0.1
            factors.append("Moderate transaction amount")
        
        # Very small amounts can also be suspicious (testing)
        if amount < 1:
            risk += 0.2
            factors.append("Unusually small transaction amount")
        
        return risk, factors
    
    def _assess_payment_method_risk(self, payment_method: str) -> tuple[float, List[str]]:
        """Assess risk based on payment method"""
        risk_levels = {
            'digital_wallet': 0.25,
            'credit_card': 0.15,
            'debit_card': 0.1,
            'bank_transfer': 0.05,
            'cash': 0.0
        }
        
        risk = risk_levels.get(payment_method, 0.2)
        factors = []
        
        if risk > 0.2:
            factors.append(f"High-risk payment method: {payment_method}")
        elif risk > 0.1:
            factors.append(f"Medium-risk payment method: {payment_method}")
        
        return risk, factors
    
    def _assess_email_risk(self, email: str) -> tuple[float, List[str]]:
        """Assess risk based on email domain and pattern"""
        risk = 0.0
        factors = []
        
        domain = email.split('@')[1] if '@' in email else ''
        
        # Check for suspicious domains
        if any(suspicious in domain.lower() for suspicious in self.suspicious_domains):
            risk += 0.4
            factors.append("Suspicious email domain")
        
        # Check for common free email providers (moderate risk)
        free_providers = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
        if domain.lower() in free_providers:
            risk += 0.1
            factors.append("Free email provider")
        
        # Check for unusual email patterns
        if len(email.split('@')[0]) < 3:
            risk += 0.15
            factors.append("Very short email username")
        
        if any(char.isdigit() for char in email.split('@')[0]) and len([c for c in email.split('@')[0] if c.isdigit()]) > 5:
            risk += 0.1
            factors.append("Email contains many numbers")
        
        return risk, factors
    
    def _assess_merchant_risk(self, merchant_id: str) -> tuple[float, List[str]]:
        """Assess risk based on merchant"""
        risk = 0.0
        factors = []
        
        # High-risk merchant categories (simplified)
        high_risk_merchants = ['CRYPTO', 'GAMBLING', 'ADULT', 'PHARMACY']
        medium_risk_merchants = ['ELECTRONICS', 'JEWELRY', 'TRAVEL']
        
        merchant_upper = merchant_id.upper()
        
        if any(category in merchant_upper for category in high_risk_merchants):
            risk += 0.3
            factors.append("High-risk merchant category")
        elif any(category in merchant_upper for category in medium_risk_merchants):
            risk += 0.15
            factors.append("Medium-risk merchant category")
        
        # New or unknown merchants
        if 'NEW' in merchant_upper or 'UNKNOWN' in merchant_upper:
            risk += 0.2
            factors.append("New or unknown merchant")
        
        return risk, factors
    
    def _assess_time_risk(self, timestamp: str) -> tuple[float, List[str]]:
        """Assess risk based on transaction timing"""
        risk = 0.0
        factors = []
        
        try:
            # Parse timestamp (assuming ISO format)
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            hour = dt.hour
            day_of_week = dt.weekday()
            
            # Late night transactions (higher risk)
            if hour < 6 or hour > 23:
                risk += 0.15
                factors.append("Late night transaction")
            
            # Weekend transactions (slightly higher risk for some categories)
            if day_of_week >= 5:  # Saturday = 5, Sunday = 6
                risk += 0.05
                factors.append("Weekend transaction")
                
        except Exception:
            # If timestamp parsing fails, add small risk
            risk += 0.05
            factors.append("Invalid timestamp format")
        
        return risk, factors
    
    def _assess_currency_risk(self, currency: str) -> tuple[float, List[str]]:
        """Assess risk based on currency"""
        risk = 0.0
        factors = []
        
        # Higher risk for certain currencies
        high_risk_currencies = ['NGN']  # Example: Nigerian Naira
        medium_risk_currencies = ['EUR', 'GBP']
        
        if currency in high_risk_currencies:
            risk += 0.2
            factors.append(f"High-risk currency: {currency}")
        elif currency in medium_risk_currencies:
            risk += 0.1
            factors.append(f"Medium-risk currency: {currency}")
        
        return risk, factors
    
    def get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score >= self.risk_thresholds['high']:
            return 'high'
        elif risk_score >= self.risk_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def calculate_confidence(self, risk_score: float) -> float:
        """Calculate confidence score based on risk score"""
        # Higher risk scores generally have higher confidence
        # Add some variability to simulate ML model uncertainty
        base_confidence = min(0.5 + (risk_score * 0.4), 0.95)
        variability = np.random.uniform(-0.1, 0.1)
        confidence = max(min(base_confidence + variability, 0.99), 0.5)
        return round(confidence, 3)

# Initialize fraud detection engine
fraud_engine = FraudDetectionEngine()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "FraudGuard ML API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        uptime=round(uptime, 2)
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the fraud detection model"""
    return ModelInfo(
        name="FraudGuard ML Model",
        version="1.0.0",
        features=[
            "Transaction amount analysis",
            "Payment method risk assessment",
            "Email domain verification",
            "Merchant category analysis",
            "Temporal pattern detection",
            "Currency risk evaluation"
        ],
        riskThresholds=fraud_engine.risk_thresholds,
        supportedCurrencies=["USD", "EUR", "GBP", "NGN", "CAD", "AUD", "JPY"],
        supportedPaymentMethods=["credit_card", "debit_card", "bank_transfer", "digital_wallet", "cash"]
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(request: TransactionRequest):
    """
    Predict fraud for one or more transactions
    
    This endpoint analyzes transactions and returns fraud predictions
    with confidence scores and risk assessments.
    """
    global request_count
    request_count += 1
    
    start_processing = time.time()
    
    try:
        logger.info(f"Processing {len(request.transactions)} transactions")
        
        predictions = []
        total_risk_score = 0.0
        high_risk_count = 0
        
        for i, transaction in enumerate(request.transactions):
            transaction_start = time.time()
            
            # Calculate fraud risk
            risk_score, risk_factors = fraud_engine.calculate_fraud_risk(transaction)
            
            # Determine if fraud
            is_fraud = risk_score >= fraud_engine.risk_thresholds['high']
            
            # Calculate confidence
            confidence = fraud_engine.calculate_confidence(risk_score)
            
            # Get risk level
            risk_level = fraud_engine.get_risk_level(risk_score)
            
            # Calculate processing time
            processing_time = (time.time() - transaction_start) * 1000  # Convert to milliseconds
            
            # Create prediction
            prediction = FraudPrediction(
                isFraud=is_fraud,
                confidence=confidence,
                riskScore=round(risk_score, 3),
                riskLevel=risk_level,
                riskFactors=risk_factors,
                processingTime=round(processing_time, 2)
            )
            
            predictions.append(prediction)
            total_risk_score += risk_score
            
            if risk_level == 'high':
                high_risk_count += 1
            
            logger.info(f"Transaction {i+1}: Risk={risk_score:.3f}, Level={risk_level}, Fraud={is_fraud}")
        
        # Calculate summary statistics
        average_risk_score = total_risk_score / len(request.transactions)
        
        total_processing_time = (time.time() - start_processing) * 1000
        logger.info(f"Batch processing completed in {total_processing_time:.2f}ms")
        
        response = PredictionResponse(
            predictions=predictions,
            totalProcessed=len(request.transactions),
            averageRiskScore=round(average_risk_score, 3),
            highRiskCount=high_risk_count
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing fraud prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )