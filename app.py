import os
import json
import base64
import asyncio
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
import websockets
from openai import AsyncOpenAI
import numpy as np
from scipy import signal

app = FastAPI()

# OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    
    # Configure session to use G.711 mulaw directly (same as Twilio)
    session_config = {
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "instructions": "Du bist Willy, ein freundlicher Assistent für ältere Menschen in Österreich. Sprich Deutsch. Halte deine Antworten kurz, klar und hilfreich. Sei geduldig und höflich.",
            "voice": "echo",
            "input_audio_format": "g711_ulaw",  # Changed to mulaw
            "output_audio_format": "g711_ulaw",  # Changed to mulaw
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 700
            },
            "temperature": 0.7
        }
    }
    
    await ws.send(json.dumps(session_config))
    
    # Wait for session confirmation
    response = await ws.recv()
    event = json.loads(response)
    if event.get("type") == "session.updated":
        print("OpenAI session configured with G.711 mulaw")
    
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
                
                # Get audio payload (already in mulaw format)
                audio_payload = data['media']['payload']
                
                # Send directly to OpenAI (no conversion needed!)
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
            
            if event_type == 'response.audio.delta':
                audio_count += 1
                if audio_count % 50 == 0:
                    print(f"Sent {audio_count} audio packets to Twilio")
                
                # OpenAI sends mulaw audio (same as Twilio expects!)
                audio_delta = event['delta']
                
                # Get current stream SID
                sid = get_stream_sid()
                if sid:
                    # Send directly to Twilio (no conversion needed!)
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
    uvicorn.run(app, host="0.0.0.0", port=5000)