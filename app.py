import os
import json
import base64
import asyncio
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
import websockets
from openai import AsyncOpenAI
import httpx  # For web search API calls
from datetime import datetime

app = FastAPI()

# OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tavily API key (get from https://tavily.com)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

async def search_web(query: str) -> str:
    """Search the web using Tavily API"""
    try:
        print(f"Searching for: {query}")
        print(f"Using Tavily API key: {TAVILY_API_KEY[:10]}..." if TAVILY_API_KEY else "NO API KEY!")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 3,
                    "include_answer": True
                },
                timeout=10.0
            )
            
            print(f"Tavily response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Tavily response data: {json.dumps(data, indent=2)[:500]}")
                
                # Get the answer summary if available
                if data.get("answer"):
                    return data["answer"]
                
                # Otherwise combine results
                results = []
                for result in data.get("results", [])[:2]:
                    content = result.get("content", "")
                    if content:
                        results.append(content)
                
                if results:
                    return " ".join(results)
                else:
                    return "Ich konnte keine Informationen dazu finden."
            else:
                error_text = response.text
                print(f"Tavily error response: {error_text}")
                return "Entschuldigung, die Suche hat nicht funktioniert."
                
    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        return "Entschuldigung, ich konnte keine aktuellen Informationen finden."

# Define the function schema for OpenAI
FUNCTION_DEFINITIONS = [
    {
        "type": "function",  # ADD THIS LINE!
        "name": "search_web",
        "description": "Durchsucht das Internet nach aktuellen Informationen. Verwende dies für Fragen über Wetter, Nachrichten, Veranstaltungen, Gottesdienstzeiten, oder andere aktuelle Informationen.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Die Suchanfrage auf Deutsch oder Englisch"
                }
            },
            "required": ["query"]
        }
    }
]

@app.post("/voice")
async def voice(request: Request):
    """Initial call handler - connects to WebSocket stream"""
    host = request.headers.get("host")
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Hans" language="de-DE">Einen Moment bitte.</Say>
    <Connect>
        <Stream url="wss://{host}/media-stream" />
    </Connect>
</Response>"""
    
    return Response(content=twiml, media_type="application/xml")

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """Handle WebSocket connection from Twilio"""
    await websocket.accept()
    print("WebSocket connection established with Twilio")
    
    # Shared state
    stream_sid = None
    openai_ws = None
    
    async def get_stream_sid():
        """Wait for stream to start and get the SID"""
        nonlocal stream_sid
        async for message in websocket.iter_text():
            data = json.loads(message)
            if data['event'] == 'start':
                stream_sid = data['start']['streamSid']
                print(f"Stream started: {stream_sid}")
                return data
    
    try:
        # Wait for stream to start
        start_data = await get_stream_sid()
        
        # Connect to OpenAI Realtime API
        openai_ws = await connect_to_openai()
        print("Connected to OpenAI Realtime API")
        
        # Create tasks for bidirectional streaming
        receive_task = asyncio.create_task(
            receive_from_twilio(websocket, openai_ws)
        )
        send_task = asyncio.create_task(
            send_to_twilio(websocket, openai_ws, lambda: stream_sid)
        )
        
        # Wait for either task to complete
        done, pending = await asyncio.wait(
            [receive_task, send_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
            
    except Exception as e:
        print(f"Error in media stream: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if openai_ws:
            await openai_ws.close()
        print("Connection closed")

async def connect_to_openai():
    """Connect to OpenAI Realtime API via WebSocket"""
    url = f"wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
    
    ws = await websockets.connect(
        url,
        additional_headers={
            "Authorization": f"Bearer {openai_client.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
    )
    
    # Configure session with function calling
    session_config = {
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "instructions": f"""Du bist Willy, ein freundlicher Assistent für ältere Menschen in Österreich. 
            Sprich Deutsch. Halte deine Antworten kurz, klar und hilfreich. Sei geduldig und höflich.
            
            Aktuelles Datum: {datetime.now().strftime('%d.%m.%Y')}
            Ort: Vorchdorf, Österreich
            
            Wenn Benutzer nach aktuellen Informationen fragen (Wetter, Veranstaltungen, Messen, Nachrichten), 
            verwende die search_web Funktion.""",
            "voice": "echo",
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 700
            },
            "temperature": 0.7,
            "tools": FUNCTION_DEFINITIONS,  # Add function definitions
            "tool_choice": "auto"
        }
    }
    
    await ws.send(json.dumps(session_config))
    
    # Wait for session confirmation
    response = await ws.recv()
    event = json.loads(response)
    if event.get("type") == "session.updated":
        print("OpenAI session configured with web search capability")
    
    # Send initial greeting
    greeting = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Sage eine kurze Begrüßung auf Deutsch und frage wie du helfen kannst."
                }
            ]
        }
    }
    await ws.send(json.dumps(greeting))
    await ws.send(json.dumps({"type": "response.create"}))
    
    return ws

async def receive_from_twilio(twilio_ws: WebSocket, openai_ws):
    """Receive audio from Twilio and forward to OpenAI"""
    audio_count = 0
    
    try:
        async for message in twilio_ws.iter_text():
            data = json.loads(message)
            
            if data['event'] == 'media':
                audio_count += 1
                if audio_count % 100 == 0:
                    print(f"Received {audio_count} audio packets from Twilio")
                
                audio_payload = data['media']['payload']
                
                audio_message = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_payload
                }
                await openai_ws.send(json.dumps(audio_message))
                
            elif data['event'] == 'stop':
                print("Stream stopped by Twilio")
                break
                
    except Exception as e:
        print(f"Error receiving from Twilio: {e}")
        import traceback
        traceback.print_exc()

async def send_to_twilio(twilio_ws: WebSocket, openai_ws, get_stream_sid):
    """Receive audio from OpenAI and forward to Twilio"""
    audio_count = 0
    
    try:
        async for message in openai_ws:
            event = json.loads(message)
            event_type = event.get('type')
            
            if event_type not in ['response.audio.delta', 'input_audio_buffer.speech_started', 'input_audio_buffer.speech_stopped']:
                print(f"OpenAI event: {event_type}")
            
            # Handle function calls
            if event_type == 'response.function_call_arguments.done':
                call_id = event['call_id']
                function_name = event['name']
                arguments = json.loads(event['arguments'])
                
                print(f"Function call: {function_name} with args: {arguments}")
                
                # Execute the function
                if function_name == "search_web":
                    search_result = await search_web(arguments['query'])
                    print(f"Search result: {search_result[:100]}...")
                    
                    # Send function result back to OpenAI
                    function_output = {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": search_result
                        }
                    }
                    await openai_ws.send(json.dumps(function_output))
                    
                    # Trigger response generation
                    await openai_ws.send(json.dumps({"type": "response.create"}))
            
            elif event_type == 'response.audio.delta':
                audio_count += 1
                if audio_count % 50 == 0:
                    print(f"Sent {audio_count} audio packets to Twilio")
                
                audio_delta = event['delta']
                
                sid = get_stream_sid()
                if sid:
                    media_message = {
                        "event": "media",
                        "streamSid": sid,
                        "media": {
                            "payload": audio_delta
                        }
                    }
                    await twilio_ws.send_json(media_message)
                
            elif event_type == 'response.audio.done':
                print("Audio response complete")
                
            elif event_type == 'conversation.item.input_audio_transcription.completed':
                transcript = event.get('transcript', '')
                print(f"User said: {transcript}")
                
            elif event_type == 'response.done':
                print("Response fully complete")
                
            elif event_type == 'error':
                print(f"OpenAI error: {event}")
                
    except Exception as e:
        print(f"Error sending to Twilio: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import uvicorn

    print(f"OpenAI API Key: {'✓ Set' if os.getenv('OPENAI_API_KEY') else '✗ Missing'}")
    print(f"Tavily API Key: {'✓ Set' if os.getenv('TAVILY_API_KEY') else '✗ Missing'}")
    
    # Test Tavily connection
    if TAVILY_API_KEY:
        print(f"Tavily API Key starts with: {TAVILY_API_KEY[:10]}...")

    uvicorn.run(app, host="0.0.0.0", port=5000)