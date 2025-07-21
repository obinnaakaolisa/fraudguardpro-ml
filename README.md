# FraudGuard ML API

A FastAPI-based fraud detection service that provides machine learning-powered fraud prediction for financial transactions.

## Features

- **Real-time Fraud Detection**: Analyze transactions and predict fraud probability
- **Batch Processing**: Handle multiple transactions in a single request
- **Risk Assessment**: Multi-factor risk scoring with detailed risk factors
- **RESTful API**: Clean, well-documented API endpoints
- **Production Ready**: Docker support, health checks, and comprehensive error handling

## Quick Start

### Using Python

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API**
   ```bash
   python main.py
   ```

3. **Access the API**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

### Using Docker

1. **Build and Run**
   ```bash
   docker-compose up --build
   ```

2. **Access the API**
   - API: http://localhost:8000

## API Endpoints

### Core Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /model/info` - Model information and capabilities
- `POST /predict` - Fraud prediction (main endpoint)

### Fraud Prediction

**Endpoint**: `POST /predict`

**Request Format**:
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
      "description": "Electronics purchase",
      "timestamp": "2024-01-15T14:30:00Z"
    }
  ]
}
```

**Response Format**:
```json
{
  "predictions": [
    {
      "isFraud": false,
      "confidence": 0.85,
      "riskScore": 0.23,
      "riskLevel": "low",
      "riskFactors": ["Moderate transaction amount"],
      "processingTime": 12.5
    }
  ],
  "totalProcessed": 1,
  "averageRiskScore": 0.23,
  "highRiskCount": 0
}
```

## Risk Assessment Factors

The fraud detection engine evaluates multiple risk factors:

### Transaction Amount
- Very high amounts (>$50,000): High risk
- High amounts (>$10,000): Medium-high risk
- Moderate amounts (>$1,000): Low-medium risk
- Very small amounts (<$1): Suspicious (testing)

### Payment Methods
- Digital wallets: Higher risk
- Credit cards: Medium risk
- Debit cards: Lower risk
- Bank transfers: Low risk
- Cash: Lowest risk

### Email Analysis
- Temporary email domains: High risk
- Free email providers: Low risk
- Short usernames: Medium risk
- Number-heavy usernames: Low risk

### Merchant Categories
- High-risk: Crypto, gambling, adult content
- Medium-risk: Electronics, jewelry, travel
- New/unknown merchants: Higher risk

### Temporal Patterns
- Late night transactions (before 6 AM, after 11 PM): Higher risk
- Weekend transactions: Slightly higher risk

### Currency Risk
- Certain currencies may have higher associated risk
- Based on regional fraud patterns

## Testing

Run the comprehensive test suite:

```bash
python test_api.py
```

The test suite includes:
- Health check validation
- Single transaction prediction
- High-risk transaction detection
- Batch processing
- Invalid data handling
- Performance testing

## Configuration

### Supported Currencies
- USD, EUR, GBP, NGN, CAD, AUD, JPY

### Supported Payment Methods
- credit_card, debit_card, bank_transfer, digital_wallet, cash

### Risk Thresholds
- Low risk: < 0.3
- Medium risk: 0.3 - 0.6
- High risk: > 0.6

## Integration with Node.js Backend

### Environment Variables
Add to your Node.js application:
```bash
PYTHON_ML_API_URL=http://localhost:8000
```

### Sample Integration Code
```javascript
const ML_API_URL = process.env.PYTHON_ML_API_URL || "http://localhost:8000";

async function callFraudDetection(transactions) {
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
    console.error('Fraud detection API call failed:', error);
    throw error;
  }
}
```

## Production Deployment

### Docker Production
```bash
# Build production image
docker build -t fraudguard-ml-api .

# Run with production settings
docker run -d \
  --name fraudguard-ml \
  -p 8000:8000 \
  --restart unless-stopped \
  fraudguard-ml-api
```

### Environment Variables
- `PYTHONPATH`: Set to `/app`
- `PYTHONDONTWRITEBYTECODE`: Set to `1`
- `PYTHONUNBUFFERED`: Set to `1`

### Health Monitoring
The API includes built-in health checks:
- HTTP endpoint: `GET /health`
- Docker health check: Automatic container health monitoring

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive API documentation with request/response examples and the ability to test endpoints directly.

## Performance

- **Single Transaction**: ~10-20ms processing time
- **Batch Processing**: ~5-10ms per transaction in batch
- **Concurrent Requests**: Supports multiple simultaneous requests
- **Memory Usage**: Lightweight, suitable for containerized deployment

## Error Handling

The API provides comprehensive error handling:
- **400**: Bad Request (malformed JSON)
- **422**: Validation Error (invalid data)
- **500**: Internal Server Error

All errors include detailed messages for debugging.

## Logging

The API includes structured logging:
- Request/response logging
- Error tracking
- Performance metrics
- Health check status

## Security Considerations

- **CORS**: Configured for cross-origin requests
- **Input Validation**: Comprehensive request validation
- **Error Handling**: Secure error messages (no sensitive data exposure)
- **Rate Limiting**: Consider implementing for production use
- **Authentication**: Can be extended with JWT or API key authentication

## Future Enhancements

- Machine learning model training pipeline
- Real-time model updates
- Advanced feature engineering
- Integration with external fraud databases
- A/B testing for model versions
- Advanced analytics and reporting