"""
DeepSeek LLM Service
Separate Railway service for local DeepSeek model processing
"""

import os
import asyncio
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

app = FastAPI(title="DeepSeek LLM Service", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
model_initialized = False

class GenerateRequest(BaseModel):
    message: str
    context: Optional[List[Dict[str, str]]] = None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7

class GenerateResponse(BaseModel):
    content: str
    model: str
    tokens_generated: int
    success: bool

@app.on_event("startup")
async def startup_event():
    """Initialize the DeepSeek model on startup"""
    global model, model_initialized
    
    if not LLAMA_CPP_AVAILABLE:
        print("WARNING: llama-cpp-python not available. Service will not function.")
        return
    
    try:
        model_path = "/app/models"
        model_name = "deepseek-coder-1.3b-instruct.Q4_K_M.gguf"
        model_file = os.path.join(model_path, model_name)
        
        # Create models directory
        os.makedirs(model_path, exist_ok=True)
        
        # Download model if it doesn't exist
        if not os.path.exists(model_file):
            print(f"Model not found at {model_file}. Please download manually or implement download logic.")
            return
        
        print(f"Loading DeepSeek model from {model_file}...")
        
        # Initialize the model
        model = Llama(
            model_path=model_file,
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )
        
        model_initialized = True
        print("DeepSeek model loaded successfully!")
        
    except Exception as e:
        print(f"Failed to initialize DeepSeek model: {e}")
        model_initialized = False

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model_initialized else "unhealthy",
        "model_loaded": model_initialized,
        "llama_cpp_available": LLAMA_CPP_AVAILABLE
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using the local DeepSeek model"""
    if not model_initialized or model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not initialized or not available"
        )
    
    try:
        # Format prompt
        prompt = format_prompt(request.message, request.context)
        
        # Generate response in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            generate_response,
            prompt,
            request.max_tokens,
            request.temperature
        )
        
        return GenerateResponse(
            content=result["content"],
            model="deepseek-coder-1.3b-instruct",
            tokens_generated=result["tokens_generated"],
            success=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

def format_prompt(message: str, context: Optional[List[Dict[str, str]]] = None) -> str:
    """Format prompt for DeepSeek model"""
    prompt = ""
    
    # Add context if provided
    if context:
        for ctx in context:
            role = ctx.get("role", "user")
            content = ctx.get("content", "")
            if role == "user":
                prompt += f"Human: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
    
    # Add current message
    prompt += f"Human: {message}\nAssistant:"
    
    return prompt

def generate_response(prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
    """Generate response using the local model (synchronous)"""
    try:
        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "Human:", "Assistant:"],
            echo=False
        )
        
        content = output["choices"][0]["text"].strip()
        
        return {
            "content": content,
            "tokens_generated": output["usage"]["completion_tokens"]
        }
        
    except Exception as e:
        raise Exception(f"Model generation failed: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "DeepSeek LLM Service",
        "status": "running",
        "model_loaded": model_initialized
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

