# Python Backend Integration Guide

This document outlines how to integrate a Python-based fraud detection backend with the FraudGuard frontend application.

## Overview

The FraudGuard frontend is designed to work with an external Python backend that provides advanced fraud detection capabilities using machine learning models. The integration follows a microservices architecture where the Node.js backend handles user management and data persistence, while the Python backend focuses on fraud prediction.

## Python Backend Requirements

### API Endpoint Specification

The Python backend should expose a POST endpoint at `/predict` that accepts transaction data and returns fraud predictions.

#### Request Format
```http
POST /predict
Content-Type: application/json
```

```json
{
  "transactions": [
    {
      "amount": 1250.00,
      "currency": "USD",
      "merchantId": "AMAZON_001", 
      "paymentMethod": "credit_card",
      "customerEmail": "alice.johnson@gmail.com",
      "ipAddress": "192.168.1.1",
      "deviceId": "DEV_001",
      "description": "Electronics purchase"
    }
  ]
}
```

#### Response Format
```json
{
  "predictions": [
    {
      "isFraud": false,
      "confidence": 0.85,
      "riskScore": 0.23,
      "riskLevel": "low"
    }
  ]
}
```

### Required Features

1. **CORS Support**: Enable cross-origin requests from the frontend application
2. **Batch Processing**: Handle multiple transactions in a single request
3. **Error Handling**: Return appropriate HTTP status codes and error messages
4. **Authentication**: Optional - can integrate with JWT tokens if needed

### Sample Python Implementation (FastAPI)

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np

app = FastAPI(title="FraudGuard ML API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Transaction(BaseModel):
    amount: float
    currency: str
    merchantId: str
    paymentMethod: str
    customerEmail: str
    ipAddress: str = None
    deviceId: str = None
    description: str = None

class TransactionRequest(BaseModel):
    transactions: List[Transaction]

class FraudPrediction(BaseModel):
    isFraud: bool
    confidence: float
    riskScore: float
    riskLevel: str

class PredictionResponse(BaseModel):
    predictions: List[FraudPrediction]

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(request: TransactionRequest):
    try:
        predictions = []
        
        for transaction in request.transactions:
            # Your ML model prediction logic here
            risk_score = calculate_fraud_risk(transaction)
            confidence = min(max(risk_score * 2, 0.5), 0.99)
            is_fraud = risk_score > 0.7
            risk_level = "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low"
            
            predictions.append(FraudPrediction(
                isFraud=is_fraud,
                confidence=confidence,
                riskScore=risk_score,
                riskLevel=risk_level
            ))
        
        return PredictionResponse(predictions=predictions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def calculate_fraud_risk(transaction: Transaction) -> float:
    """
    Placeholder fraud detection algorithm.
    Replace with your actual ML model.
    """
    risk = 0.0
    
    # Amount-based risk
    if transaction.amount > 10000:
        risk += 0.4
    elif transaction.amount > 5000:
        risk += 0.3
    elif transaction.amount > 1000:
        risk += 0.2
    else:
        risk += 0.1
    
    # Payment method risk
    if transaction.paymentMethod == "digital_wallet":
        risk += 0.2
    elif transaction.paymentMethod == "bank_transfer":
        risk += 0.1
    else:
        risk += 0.15
    
    # Email domain risk
    if transaction.customerEmail and "@" in transaction.customerEmail:
        domain = transaction.customerEmail.split("@")[1]
        if "tempmail" in domain or "10minute" in domain:
            risk += 0.3
    
    # Add some randomness to simulate ML model variability
    risk += np.random.uniform(0, 0.1)
    
    return min(risk, 1.0)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Integration with FraudGuard Frontend

### Environment Configuration

Add the Python backend URL to your environment variables:

```bash
PYTHON_ML_API_URL=http://localhost:8000
```

### Frontend Integration

The FraudGuard frontend can be extended to use the Python backend by modifying the transaction processing endpoints in `server/routes.ts`:

```typescript
// Add to server/routes.ts
const ML_API_URL = process.env.PYTHON_ML_API_URL || "http://localhost:8000";

async function callPythonML(transactions: any[]) {
  try {
    const response = await fetch(`${ML_API_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ transactions }),
    });
    
    if (!response.ok) {
      throw new Error(`ML API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Python ML API call failed:', error);
    // Fallback to local risk calculation
    return null;
  }
}
```

### Deployment Considerations

1. **Docker Deployment**: Both services can be containerized and deployed together
2. **Microservices**: Deploy Python backend as a separate service
3. **Load Balancing**: Use a reverse proxy for production deployments
4. **Security**: Implement proper authentication between services

### Model Training Data Format

The Python backend should expect transaction features in this format for model training:

```json
{
  "amount": 1250.00,
  "currency_usd": 1,
  "currency_eur": 0, 
  "currency_gbp": 0,
  "currency_ngn": 0,
  "payment_method_credit_card": 1,
  "payment_method_debit_card": 0,
  "payment_method_bank_transfer": 0,
  "payment_method_digital_wallet": 0,
  "email_domain_suspicious": 0,
  "amount_category": 2,
  "hour_of_day": 14,
  "day_of_week": 3
}
```

## Testing

Use the sample data provided in the application to test the Python backend integration:

1. Load sample data using the "Load Sample Data" button in the dashboard
2. Verify predictions are being returned correctly
3. Test error handling with invalid requests
4. Validate CORS functionality from the frontend

## Next Steps

1. Implement the Python FastAPI backend
2. Add environment variable for ML API URL
3. Modify transaction processing to call Python API
4. Train ML models with historical transaction data
5. Implement model versioning and A/B testing capabilities