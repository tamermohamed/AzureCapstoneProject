import requests
import json

class ConsumeEndPoint:

    def SendRequest(scoring_uri,key):

            # Two sets of data to score, so we get two results back
            data = {"data":
                    [
                      {
                        "sex": 1,
                        "age": 34.5,
                        "sibSp": 0,
                        "parch": 0,
                        "embarked": 3,
                        "pclass" : 1
                      }
                  ]
                }
            # Convert to JSON string
            input_data = json.dumps(data)
            with open("data.json", "w") as _f:
                _f.write(input_data)

            # Set the content type
            headers = {'Content-Type': 'application/json'}
            # If authentication is enabled, set the authorization header
            headers['Authorization'] = f'Bearer {key}'

            # Make the request and display the response
            resp = requests.post(scoring_uri, input_data, headers=headers)
            print(resp.json())


