import requests
import json

def test_recommendations():
    # API endpoint
    url = 'http://localhost:5000/api/recommend'
    
    # Sample movie list
    payload = {
        "movies": [
            "Toy Story (1995)",
            "Jumanji (1995)",
            "Grumpier Old Men (1995)",
            "Waiting to Exhale (1995)",
            "Father of the Bride Part II (1995)"
        ]
    }
    
    # Headers
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        # Make the POST request
        response = requests.post(url, json=payload, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Print the recommendations
            recommendations = response.json()['recommendations']
            print("\nRecommended Movies:")
            for i, movie in enumerate(recommendations, 1):
                print(f"{i}. {movie}")
        else:
            # Print error message
            print(f"Error: {response.status_code}")
            print(response.json())
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    print("Testing movie recommendations API...")
    test_recommendations()
