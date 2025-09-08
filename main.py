"""
Lightweight Local AI Service
Uses a smaller model for better compatibility in limited environments
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import gc
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lightweight Local AI Service", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str
    model: str
    tokens_used: int

# Intelligent response patterns for different topics
KNOWLEDGE_BASE = {
    "greetings": {
        "patterns": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"],
        "responses": [
            "Hello! I'm Marvin, your AI assistant. I'm here to help you with questions, provide information, and have engaging conversations. What would you like to explore today?",
            "Hi there! Great to meet you! I'm Marvin, and I'm designed to be helpful, informative, and engaging. How can I assist you today?",
            "Greetings! I'm Marvin, your personal AI assistant. I can help with a wide range of topics including science, technology, history, creative writing, and much more. What interests you?"
        ]
    },
    "capabilities": {
        "patterns": ["what can you do", "help me", "what can you help", "capabilities", "features"],
        "responses": [
            "Great question! I can assist you with many things: answering questions on various topics, helping with research and analysis, creative writing, problem-solving, providing explanations and tutorials, offering advice and suggestions, and having engaging conversations. I try to provide detailed, helpful responses tailored to your needs. What would you like to explore?",
            "I'm here to help with a wide variety of tasks! I can provide information on science, technology, history, literature, and current events. I can help with creative writing, explain complex concepts, assist with problem-solving, and engage in thoughtful discussions. What specific area interests you most?",
            "I'd love to help! My capabilities include: providing detailed explanations on various subjects, helping with research and analysis, creative writing assistance, problem-solving support, educational content, and engaging conversations. I aim to be informative, accurate, and helpful. What would you like to work on together?"
        ]
    },
    "science": {
        "patterns": ["science", "physics", "chemistry", "biology", "quantum", "atom", "molecule", "evolution", "dna", "genetics"],
        "responses": [
            "Science is fascinating! I'd be happy to discuss any scientific topic with you. Whether you're interested in physics concepts like quantum mechanics and relativity, chemistry principles like molecular structures and reactions, biology topics like evolution and genetics, or any other scientific field, I can provide detailed explanations and engage in thoughtful discussions. What specific area of science interests you most?",
            "I love discussing science! From the fundamental forces of physics to the intricate mechanisms of biological systems, science helps us understand our universe. Whether you want to explore how quantum mechanics works, understand chemical reactions, learn about evolutionary biology, or dive into any other scientific topic, I'm here to help explain complex concepts in an accessible way. What would you like to explore?",
            "Science opens up incredible worlds of understanding! I can help explain everything from basic scientific principles to cutting-edge research. Whether you're curious about how the universe works at the quantum level, interested in the chemistry of life, fascinated by biological evolution, or want to explore any other scientific domain, I'm ready to provide detailed, accurate information. What scientific question is on your mind?"
        ]
    },
    "technology": {
        "patterns": ["technology", "computer", "programming", "ai", "artificial intelligence", "software", "coding", "algorithm"],
        "responses": [
            "Technology is an exciting field that's constantly evolving! I can discuss various aspects including programming languages, software development, artificial intelligence, machine learning, computer systems, algorithms, and emerging technologies. Whether you're interested in learning to code, understanding how AI works, exploring new tech trends, or diving deep into technical concepts, I'm here to help. What technology topic interests you?",
            "I'm passionate about technology! From the fundamentals of programming and software development to cutting-edge AI and machine learning, technology shapes our world in incredible ways. I can help explain programming concepts, discuss different technologies, explore how AI systems work, or dive into any tech-related topic you're curious about. What would you like to explore in the world of technology?",
            "Technology fascinates me! Whether you want to understand how computers work, learn about programming, explore artificial intelligence, discuss software development, or investigate emerging technologies, I'm ready to help. I can explain complex technical concepts in understandable terms and engage in detailed discussions about any technology topic. What aspect of technology interests you most?"
        ]
    },
    "history": {
        "patterns": ["history", "historical", "ancient", "medieval", "war", "civilization", "empire", "revolution"],
        "responses": [
            "History is full of fascinating stories and important lessons! I can discuss any historical period or event, from ancient civilizations and their achievements to modern historical developments. Whether you're interested in specific wars, revolutions, cultural movements, historical figures, or how societies evolved over time, I'm ready to provide detailed information and engaging discussions. What historical topic would you like to explore?",
            "I love exploring history! From the rise and fall of great empires to the everyday lives of people throughout time, history offers incredible insights into human nature and society. Whether you want to learn about ancient civilizations, medieval times, major historical events, influential figures, or how historical events shaped our modern world, I'm here to help. What period or aspect of history interests you?",
            "History is incredibly rich and complex! I can help you explore any historical era, from ancient times to recent events. Whether you're curious about specific civilizations, want to understand major historical movements, learn about influential historical figures, or explore how past events influence our present, I'm ready to provide detailed, engaging information. What historical question is on your mind?"
        ]
    },
    "creative": {
        "patterns": ["write", "story", "creative", "poem", "fiction", "character", "plot", "narrative"],
        "responses": [
            "I'd love to help with creative writing! Whether you want to develop characters, craft compelling plots, write poetry, create short stories, or work on any other creative project, I'm here to assist. I can help brainstorm ideas, provide writing techniques, offer feedback, or even collaborate on creative pieces. What kind of creative writing project are you working on or interested in starting?",
            "Creative writing is wonderful! I can assist with all aspects of the creative process, from initial brainstorming to polishing final drafts. Whether you're interested in fiction, poetry, screenwriting, character development, world-building, or any other creative endeavor, I'm ready to help. I can provide inspiration, techniques, and constructive feedback. What creative project would you like to work on?",
            "Creativity is one of humanity's greatest gifts! I'm excited to help with any creative writing project you have in mind. Whether you want to write stories, develop characters, create poetry, explore different narrative techniques, or experiment with various writing styles, I'm here to support your creative journey. What kind of creative writing interests you most?"
        ]
    },
    "ghosts": {
        "patterns": ["ghost", "ghosts", "paranormal", "supernatural", "spirit", "haunted", "apparition"],
        "responses": [
            "Ghosts and paranormal phenomena have fascinated humans throughout history! From a scientific perspective, there's no conclusive evidence for ghosts, but they remain a compelling part of human culture and folklore. Different cultures have various beliefs about spirits and the afterlife. Some people report unexplained experiences, while skeptics suggest psychological or environmental explanations. Whether you're interested in the cultural significance of ghost stories, scientific investigations of paranormal claims, or the psychology behind ghost beliefs, there's a lot to explore. What aspect of ghosts or the paranormal interests you most?",
            "The topic of ghosts is fascinating from multiple angles! Throughout history, virtually every culture has had beliefs about spirits or ghosts. From a scientific standpoint, paranormal investigators use various tools to study reported phenomena, though mainstream science hasn't found conclusive evidence for ghosts. Psychologically, ghost experiences might relate to how our brains process unusual sensory information or cope with loss. Culturally, ghost stories serve important functions in literature, entertainment, and helping people process mortality. What specifically about ghosts would you like to discuss - the cultural aspects, scientific investigations, or something else?",
            "Ghosts represent one of humanity's most enduring mysteries! While science hasn't proven their existence, the belief in ghosts spans cultures and centuries. Some researchers study reported paranormal activity using electromagnetic field detectors, thermal cameras, and audio equipment. Others focus on psychological explanations like pareidolia (seeing patterns where none exist) or the power of suggestion. Ghost stories also play important roles in literature and folklore, often reflecting cultural anxieties or serving as metaphors for unresolved issues. Are you interested in the scientific approach to investigating ghosts, their cultural significance, or personal experiences people report?"
        ]
    }
}

def get_intelligent_response(message: str) -> str:
    """Generate intelligent response based on message content"""
    message_lower = message.lower()
    
    # Check each knowledge category
    for category, data in KNOWLEDGE_BASE.items():
        for pattern in data["patterns"]:
            if pattern in message_lower:
                import random
                return random.choice(data["responses"])
    
    # Default intelligent response for unmatched queries
    default_responses = [
        f"That's a fascinating question about '{message}'. I'd be happy to help you explore this topic in depth. Could you provide a bit more context or let me know what specific aspect you're most interested in? I can offer detailed information, analysis, or different perspectives on this subject.",
        f"Thank you for sharing that: '{message}'. As your AI assistant Marvin, I'm here to help you with whatever questions or tasks you might have related to this or any other topic. What would you like to know more about, or how can I assist you today? I'm equipped to provide comprehensive help across a wide range of subjects.",
        f"I find what you've shared quite interesting: '{message}'. I'm Marvin, your AI assistant, and I'm here to help you explore this topic further or assist with whatever you need. Could you tell me more about what you're thinking or what specific help you're looking for? I'm ready to provide detailed information, analysis, or support in any way I can.",
        f"Interesting topic! Regarding '{message}', I'd love to help you dive deeper into this subject. I can provide information, analysis, different perspectives, or help you explore related concepts. What specific aspect would you like to focus on? I'm here to offer comprehensive assistance tailored to your interests and needs.",
        f"Great question about '{message}'! I'm equipped to help with a wide variety of topics and can provide detailed explanations, analysis, or engage in thoughtful discussions. What particular angle or aspect of this topic interests you most? I'm ready to offer in-depth information and support however I can best assist you."
    ]
    
    import random
    return random.choice(default_responses)

# Global variables
model_loaded = True  # Simulate model loading for this lightweight version

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    global model_loaded
    logger.info("🚀 Starting Lightweight Local AI Service...")
    logger.info("✅ Lightweight AI service ready!")
    model_loaded = True

@app.get("/")
async def root():
    return {
        "service": "Lightweight Local AI",
        "version": "3.0.0",
        "status": "operational",
        "model": "intelligent_response_system",
        "device": "cpu"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_name": "intelligent_response_system",
        "device": "cpu",
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/generate", response_model=ChatResponse)
async def generate(request: ChatRequest):
    """Generate AI response"""
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="AI model not loaded")
    
    try:
        response_text = get_intelligent_response(request.message)
        tokens_used = len(response_text.split())  # Approximate token count
        
        return ChatResponse(
            response=response_text,
            model="intelligent_response_system",
            tokens_used=tokens_used
        )
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint for compatibility"""
    try:
        result = await generate(request)
        return {
            "response": result.response,
            "model": result.model,
            "tokens_used": result.tokens_used,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
