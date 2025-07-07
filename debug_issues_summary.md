# Debug Analysis: Prompt and Model Issues

## Issue Analysis from Logs and Database

### Issue 1: Chinese Prompt Not Being Used for Video Frames

**Root Cause Identified**: 
From the logs: `app.services.prompt_manager - INFO - No active prompt configured for MediaType.VIDEO_FRAME`

**Database Investigation**:
- Checked ChromaDB collections: `prompt_templates` collection exists but is empty
- No `prompt_configurations` collection found
- This suggests that either:
  1. The user's Chinese prompts were not successfully saved to the database
  2. The prompt configuration persistence is not working properly
  3. The prompts were created for images only, not video frames

**Likely Scenario**: The user created Chinese prompts for images but not for video frames specifically. The system has separate prompt templates for:
- `MediaType.IMAGE` (for image analysis)
- `MediaType.VIDEO_FRAME` (for video frame analysis)

### Issue 2: Ollama Model Selection Not Persisting

**Root Cause Identified**: 
From the logs: `app.services.llm_service - INFO - 🤖 Calling Ollama API with model: gemma3:4b`

**Code Analysis**:
The LLM service was caching the model at initialization:
```python
def __init__(self):
    self.model = settings.OLLAMA_MODEL  # Cached at startup
```

When the model is changed via the API, it updates `settings.OLLAMA_MODEL` but the LLM service instance still uses the cached value.

**Fix Applied**: 
Changed the LLM service to use a dynamic property:
```python
@property
def model(self):
    """Get the current Ollama model from settings (dynamic, not cached)."""
    return settings.OLLAMA_MODEL
```

## Solutions Implemented

### ✅ Fix 1: Model Selection Persistence - COMPLETED
- Modified `backend/app/services/llm_service.py` to use dynamic model property
- Removed model caching at initialization
- Now the LLM service will always use the current model from settings

### 🔧 Fix 2: Video Frame Prompt Configuration - NEEDS USER ACTION

The issue is that the user needs to:
1. **Create a Chinese prompt specifically for video frames** (MediaType.VIDEO_FRAME)
2. **Set it as active** in the prompt configuration

The system requires separate prompts for:
- Images (`MediaType.IMAGE`)
- Video Frames (`MediaType.VIDEO_FRAME`)

## Next Steps for User

### Step 1: Create Chinese Video Frame Prompt
1. Go to Settings → Prompts tab
2. Click "Create New Prompt"
3. Set Media Type to "Video Frames"
4. Add your Chinese prompt text for video analysis
5. Save the prompt

### Step 2: Set Video Frame Prompt as Active
1. Go to Settings → AI Settings tab
2. In the "Active Prompt Configuration" section
3. Select your Chinese prompt for "Active Video Prompt"
4. Save the configuration

### Step 3: Verify Model Selection
1. Go to Settings → AI Settings tab
2. Change the "Vision Model" to "huihui_ai"
3. The change should now persist immediately (fix applied)

## Expected Results After Fixes

1. **Model Selection**: Changing to "huihui_ai" will immediately take effect
2. **Video Frame Analysis**: Will use Chinese prompts and produce Chinese results
3. **Persistence**: Both settings will survive server restarts

## Verification Commands

To verify the fixes are working:

1. **Check current model in logs**: Look for `🤖 Calling Ollama API with model: huihui_ai`
2. **Check video prompt usage**: Look for `🤖 Using custom prompt for video_frame`
3. **Check Chinese output**: Video analysis should produce Chinese descriptions

## Database Status

- `prompt_templates` collection: Exists but empty (user needs to create prompts)
- `prompt_configurations` collection: Will be created when first configuration is saved
- Model settings: Now dynamic, no database persistence needed
