# Model Selection Refresh - Complete Working Solution

## Problem Summary

The LLM service model selection was not updating during video analysis operations. Despite updating the model through the UI to "huihui_ai/gemma3-abliterated:4b", the system continued using the old model "gemma3:4b" during video frame analysis, causing errors when the old model was deleted from Ollama.

## Root Cause Analysis

The issue had **two parts**:

### Part 1: LLM Service Singleton Not Refreshing
- LLM service singleton was initialized once with original model
- When model settings were updated via API, the singleton continued using old configuration
- No mechanism existed to refresh the singleton when settings changed

### Part 2: IndexingService Caching Old LLM Instance  
- **Critical Issue**: IndexingService was caching an LLM service instance at initialization time
- `self.llm_service = LLMService()` in `__init__` created a reference to the old instance
- Even after singleton refresh, IndexingService kept using the cached old instance
- This is why video analysis operations still used the old model

## Complete Solution Implemented

### 1. Enhanced LLM Service Singleton with Refresh Capability

**File**: `backend/app/services/llm_service.py`

```python
def __init__(self):
    # Track the model we were initialized with for refresh detection
    self._initialized_model = settings.OLLAMA_MODEL
    # ... existing initialization

@classmethod
def refresh_if_model_changed(cls):
    """Check if the model has changed and refresh the singleton if needed."""
    if cls._instance is not None and hasattr(cls._instance, '_initialized_model'):
        current_model = settings.OLLAMA_MODEL
        initialized_model = cls._instance._initialized_model
        
        if current_model != initialized_model:
            logger.info(f"🔄 Model changed from '{initialized_model}' to '{current_model}' - refreshing LLM service")
            cls._instance._refresh_internal()

def _refresh_internal(self):
    """Internal method to refresh the LLM service configuration."""
    # Update client configuration
    self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)
    self.timeout = settings.OLLAMA_TIMEOUT
    
    # Update the tracked model
    old_model = self._initialized_model
    self._initialized_model = settings.OLLAMA_MODEL
    
    # Test connection with new model
    self._test_connection()
    
    logger.info(f"🔄 LLM service refreshed: '{old_model}' → '{self._initialized_model}'")
```

### 2. API Integration for Automatic Refresh

**File**: `backend/app/api/config_api.py`

```python
if config.ollama_model is not None:
    settings.OLLAMA_MODEL = config.ollama_model
    
    # Refresh LLM service if model changed
    try:
        from app.services.llm_service import LLMService
        LLMService.refresh_if_model_changed()
        logger.info("🔄 LLM service refresh triggered after model update")
    except Exception as e:
        logger.warning(f"Failed to refresh LLM service: {e}")
```

### 3. **CRITICAL FIX**: IndexingService Dynamic LLM Access

**File**: `backend/app/services/indexing.py`

**Before (Broken)**:
```python
def __init__(self):
    self.llm_service = LLMService()  # Cached old instance!
```

**After (Fixed)**:
```python
def __init__(self):
    # Don't cache LLM service - get fresh instance each time
    
@property
def llm_service(self):
    """Get the current LLM service instance (always fresh to pick up config changes)."""
    return LLMService()
```

This ensures IndexingService always gets the current singleton instance, which picks up any refresh changes.

## Testing and Verification

### Server Log Evidence

The fix is confirmed by server logs showing the complete flow:

```
2025-07-07 05:17:50,819 - app.api.config_api - INFO - 🔍 Updating OLLAMA_MODEL from 'gemma3:4b' to 'huihui_ai/gemma3-abliterated:4b'
2025-07-07 05:17:50,819 - app.services.llm_service - INFO - 🔄 Model changed from 'gemma3:4b' to 'huihui_ai/gemma3-abliterated:4b' - refreshing LLM service
2025-07-07 05:17:50,835 - app.services.llm_service - INFO - 🔄 LLM service refreshed: 'gemma3:4b' → 'huihui_ai/gemma3-abliterated:4b'
2025-07-07 05:17:50,838 - app.services.llm_service - INFO - 🎬 Starting video analysis for: [video file]
2025-07-07 05:17:56,987 - app.services.llm_service - INFO - 🤖 Calling Ollama API with model: huihui_ai/gemma3-abliterated:4b
```

**Key Evidence**: The last line shows video analysis is now using the updated model!

### Test Results

- ✅ **Model Configuration Update**: API successfully updates model settings
- ✅ **Singleton Refresh**: LLM service singleton picks up changes immediately  
- ✅ **IndexingService Integration**: IndexingService gets fresh LLM instances
- ✅ **Video Analysis**: Video frame analysis uses the updated model
- ✅ **No Server Restart Required**: Changes take effect immediately

## Files Modified

1. **`backend/app/services/llm_service.py`**:
   - Added `_initialized_model` tracking
   - Added `refresh_if_model_changed()` class method
   - Added `_refresh_internal()` instance method

2. **`backend/app/api/config_api.py`**:
   - Added refresh trigger in `update_llm_config()`
   - Added refresh trigger in `reset_llm_config()`

3. **`backend/app/services/indexing.py`**:
   - **CRITICAL**: Removed cached LLM service instance
   - Added dynamic `llm_service` property for fresh instances

## Key Benefits

1. **Immediate Effect**: Model changes take effect immediately without server restart
2. **Complete Coverage**: All analysis operations (image, video, reindexing) use updated model
3. **Singleton Preservation**: Maintains singleton pattern benefits while adding refresh capability
4. **Automatic Triggering**: Refresh happens automatically when settings are updated via API
5. **Error Handling**: Graceful handling of refresh failures without breaking API calls
6. **Backward Compatibility**: Existing functionality remains unchanged

## Usage

The fix is completely automatic. When users update the model through the UI:

1. Frontend sends API request to update model
2. Backend updates settings and triggers singleton refresh
3. IndexingService gets fresh LLM service instances via property
4. All subsequent analysis operations use the new model
5. No server restart required

## Verification Commands

To verify the fix is working:

```bash
# Start the server
cd backend && python -m uvicorn app.main:app --reload

# Trigger video analysis after model change
python test_real_video_analysis.py

# Check server logs for:
# 🔄 Model changed from X to Y - refreshing LLM service
# 🔄 LLM service refreshed: X → Y  
# 🤖 Calling Ollama API with model: [new model]
```

## Conclusion

The model selection refresh mechanism is now fully functional for all analysis operations, including video frame analysis. The critical fix was ensuring IndexingService doesn't cache old LLM service instances, allowing it to always get the refreshed singleton. Users can now change models through the UI and the changes take effect immediately for all video analysis operations.
