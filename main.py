from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import google.generativeai as genai
import json
import requests
from typing import Literal, List
from datetime import datetime, timezone
import boto3 # Import the AWS SDK
from mangum import Mangum # Import Mangum

# --- Configuration and Setup ---
# We no longer use .env files; secrets will be handled by AWS
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
CHAT_HISTORY_TABLE_NAME = os.getenv("CHAT_HISTORY_TABLE_NAME")

# Basic validation
if not all([GOOGLE_API_KEY, WEATHER_API_KEY, CHAT_HISTORY_TABLE_NAME]):
    print("WARNING: Missing environment variables. This is expected for local testing.")

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize DynamoDB client
dynamodb_client = None
if CHAT_HISTORY_TABLE_NAME:
    dynamodb_client = boto3.resource('dynamodb')
    chat_history_table = dynamodb_client.Table(CHAT_HISTORY_TABLE_NAME)

# --- FastAPI App Initialization and CORS ---
app = FastAPI(title="Gemini Weather Chatbot with History")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for simplicity, can be tightened later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Tool Definition (Weather Function) ---
def get_weather(location: str, unit: Literal["celsius", "fahrenheit"] = "celsius"):
    """Gets the current weather for a specified location."""
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": location, "appid": WEATHER_API_KEY, "units": "metric" if unit == "celsius" else "imperial"}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        return json.dumps({ "location": data["name"], "temperature": data["main"]["temp"], "unit": unit, "conditions": data["weather"][0]["description"] })
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Failed to get weather data: {str(e)}"})

# --- Database Helper Functions ---
def save_message(user_id: str, sender: str, message: str):
    """Saves a message to the DynamoDB table."""
    if not chat_history_table: return
    timestamp = datetime.now(timezone.utc).isoformat()
    chat_history_table.put_item(
        Item={ 'UserID': user_id, 'Timestamp': timestamp, 'Sender': sender, 'Message': message }
    )

def get_recent_history(user_id: str, limit: int = 5) -> List[dict]:
    """Gets the last N messages for a user from DynamoDB."""
    if not chat_history_table: return []
    response = chat_history_table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key('UserID').eq(user_id),
        ScanIndexForward=False, Limit=limit
    )
    return sorted(response.get('Items', []), key=lambda x: x['Timestamp'])

# --- API Endpoints ---
class ChatRequest(BaseModel):
    userId: str
    message: str

@app.post("/chat")
async def chat_with_gemini(request: ChatRequest):
    if not request.message or not request.userId:
        raise HTTPException(status_code=400, detail="userId and message cannot be empty")
    
    save_message(request.userId, 'USER', request.message)
    history = get_recent_history(request.userId)
    
    model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest', tools=[get_weather])
    chat = model.start_chat(history=[{"role": "user" if item['Sender'] == 'USER' else "model", "parts": [{"text": item['Message']}]} for item in history])
    response = chat.send_message(request.message)

    try:
        part = response.candidates[0].content.parts[0]
        if part.function_call:
            function_call = part.function_call
            function_response_data = get_weather(location=function_call.args.get('location'))
            api_response = chat.send_message(
                genai.protos.Part(function_response={'name': 'get_weather', 'response': {'result': function_response_data}})
            )
            response_text = api_response.text
        else:
            response_text = response.text
    except (IndexError, AttributeError):
        response_text = response.text
        
    save_message(request.userId, 'BOT', response_text)
    return {"response": response_text}

@app.get("/history/{user_id}")
async def get_full_history(user_id: str):
    return {"history": get_recent_history(user_id, limit=100)}

@app.get("/")
async def root():
    return {"message": "Backend is running"}

# This handler is used by Mangum to run the app on AWS Lambda
handler = Mangum(app)

