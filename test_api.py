"""
Test script for the FraudGuard ML API
"""
import requests
import json
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_info_endpoint():
    """Test the model info endpoint"""
    print("\nTesting model info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_single_transaction():
    """Test fraud prediction with a single transaction"""
    print("\nTesting single transaction prediction...")
    
    transaction_data = {
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
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=transaction_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_high_risk_transaction():
    """Test fraud prediction with a high-risk transaction"""
    print("\nTesting high-risk transaction prediction...")
    
    transaction_data = {
        "transactions": [
            {
                "amount": 75000.00,
                "currency": "NGN",
                "merchantId": "CRYPTO_EXCHANGE_001",
                "paymentMethod": "digital_wallet",
                "customerEmail": "user123@tempmail.org",
                "ipAddress": "10.0.0.1",
                "deviceId": "UNKNOWN_DEVICE",
                "description": "Cryptocurrency purchase",
                "timestamp": "2024-01-15T02:30:00Z"
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=transaction_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        # Check if it's correctly identified as high risk
        if result["predictions"][0]["riskLevel"] == "high":
            print("✅ High-risk transaction correctly identified!")
        else:
            print("⚠️ High-risk transaction not identified as high risk")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_batch_transactions():
    """Test fraud prediction with multiple transactions"""
    print("\nTesting batch transaction prediction...")
    
    transaction_data = {
        "transactions": [
            {
                "amount": 50.00,
                "currency": "USD",
                "merchantId": "COFFEE_SHOP_001",
                "paymentMethod": "credit_card",
                "customerEmail": "john.doe@company.com",
                "description": "Coffee purchase"
            },
            {
                "amount": 15000.00,
                "currency": "EUR",
                "merchantId": "JEWELRY_STORE_001",
                "paymentMethod": "digital_wallet",
                "customerEmail": "buyer@10minutemail.com",
                "description": "Luxury watch purchase"
            },
            {
                "amount": 200.00,
                "currency": "GBP",
                "merchantId": "BOOKSTORE_001",
                "paymentMethod": "debit_card",
                "customerEmail": "reader@gmail.com",
                "description": "Book purchase"
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=transaction_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        print(f"\nSummary:")
        print(f"Total processed: {result['totalProcessed']}")
        print(f"Average risk score: {result['averageRiskScore']}")
        print(f"High risk count: {result['highRiskCount']}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_invalid_data():
    """Test API with invalid data"""
    print("\nTesting invalid data handling...")
    
    # Test with invalid currency
    invalid_data = {
        "transactions": [
            {
                "amount": 100.00,
                "currency": "INVALID",
                "merchantId": "TEST_001",
                "paymentMethod": "credit_card",
                "customerEmail": "test@example.com"
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Should return 422 for validation error
        return response.status_code == 422
    except Exception as e:
        print(f"Error: {e}")
        return False

def run_performance_test():
    """Run a simple performance test"""
    print("\nRunning performance test...")
    
    # Create 10 transactions
    transactions = []
    for i in range(10):
        transactions.append({
            "amount": 100.00 + (i * 50),
            "currency": "USD",
            "merchantId": f"MERCHANT_{i:03d}",
            "paymentMethod": "credit_card",
            "customerEmail": f"user{i}@example.com",
            "description": f"Test transaction {i}"
        })
    
    transaction_data = {"transactions": transactions}
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/predict",
            json=transaction_data,
            headers={"Content-Type": "application/json"}
        )
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        print(f"Status: {response.status_code}")
        print(f"Total processing time: {processing_time:.2f}ms")
        print(f"Average per transaction: {processing_time/10:.2f}ms")
        
        if response.status_code == 200:
            result = response.json()
            avg_individual_time = sum(p["processingTime"] for p in result["predictions"]) / len(result["predictions"])
            print(f"Average individual processing time: {avg_individual_time:.2f}ms")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting FraudGuard ML API Tests")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Model Info", test_model_info_endpoint),
        ("Single Transaction", test_single_transaction),
        ("High Risk Transaction", test_high_risk_transaction),
        ("Batch Transactions", test_batch_transactions),
        ("Invalid Data", test_invalid_data),
        ("Performance Test", run_performance_test)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    print(f"\n{'='*50}")
    print("📊 Test Results Summary:")
    print(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! The API is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the API implementation.")

if __name__ == "__main__":
    main()