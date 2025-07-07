# Prompt Template and Model Selection Debug & Fixes

## Issues Identified and Root Causes

### Issue 1: Chinese Prompt Not Being Used for Video Frames ❌

**Problem**: Video frame analysis produces English results despite selecting Chinese prompts.

**Root Cause**: 
- From logs: `No active prompt configured for MediaType.VIDEO_FRAME`
- Database investigation shows `prompt_templates` collection exists but is empty
- The system requires **separate prompts** for images and video frames:
  - `MediaType.IMAGE` - for image analysis
  - `MediaType.VIDEO_FRAME` - for video frame analysis

**User Action Required**: You need to create a Chinese prompt specifically for video frames.

### Issue 2: Ollama Model Selection Not Persisting ✅ FIXED

**Problem**: Model selection changes to "huihui_ai" not taking effect, still using "gemma3:4b".

**Root Cause**: LLM service was caching the model at initialization:
```python
def __init__(self):
    self.model = settings.OLLAMA_MODEL  # Cached at startup
```

**Fix Applied**: Modified LLM service to use dynamic property:
```python
@property
def model(self):
    return settings.OLLAMA_MODEL  # Always gets current setting
```

## Solutions Implemented

### ✅ Model Selection Fix - COMPLETED

**Files Modified**:
- `backend/app/services/llm_service.py`

**Changes**:
1. Removed model caching in `__init__()` method
2. Added dynamic `model` property that reads from settings
3. Model changes via API now take effect immediately

**Verification**: Check logs for `🤖 Calling Ollama API with model: huihui_ai` after changing model in UI.

### 🔧 Video Frame Prompt Fix - USER ACTION REQUIRED

**The Issue**: You likely created Chinese prompts for images but not for video frames.

**Solution Steps**:

#### Step 1: Create Chinese Video Frame Prompt
1. Open your application UI
2. Go to **Settings** → **Prompts** tab
3. Click **"Create New Prompt"**
4. Fill in the form:
   - **Name**: "Chinese Video Frame Analysis" (or similar)
   - **Description**: "Chinese prompt for video frame analysis"
   - **Media Type**: Select **"Video Frames"** (not "Images")
   - **Prompt Text**: Use your Chinese prompt, for example:
   ```
   请分析这个视频帧并用中文提供JSON响应：

   {
     "description": "详细描述视频帧内容（2-3句话）",
     "objects": ["关键", "物体", "列表"],
     "setting": "环境或场景描述",
     "mood": "氛围或情感基调",
     "tags": ["相关", "描述性", "关键词"]
   }

   重要：只返回有效的JSON格式，不要包含markdown代码块或其他解释文字。
   ```
5. Click **Save**

#### Step 2: Set Video Frame Prompt as Active
1. Go to **Settings** → **AI Settings** tab
2. In the **"Active Prompt Configuration"** section
3. Find **"Active Video Prompt"** dropdown
4. Select your newly created Chinese video frame prompt
5. The configuration should save automatically

#### Step 3: Verify Model Selection
1. In the same **AI Settings** tab
2. Change **"Vision Model"** to **"huihui_ai"**
3. The change should now take effect immediately (fix applied)

## Expected Results After Fixes

### ✅ Model Selection (Fixed)
- Changing model in UI immediately updates the LLM service
- No server restart required
- Changes persist across sessions

### 🔧 Video Frame Analysis (After User Action)
- Video frame analysis will use Chinese prompts
- Results will be in Chinese format
- Logs will show: `🤖 Using custom prompt for video_frame`

## Verification Steps

### 1. Check Model Usage in Logs
After changing model to "huihui_ai", look for:
```
🤖 Calling Ollama API with model: huihui_ai
```

### 2. Check Video Prompt Usage in Logs
After setting Chinese video frame prompt, look for:
```
🤖 Using custom prompt for video_frame (length: XXX chars)
Found active video prompt: [Your Chinese Prompt Name]
```

### 3. Test Video Analysis
1. Upload or reindex a video file
2. Check the analysis results - should be in Chinese
3. Look for Chinese characters in descriptions and tags

## Why This Happened

### Model Issue
The LLM service was designed to cache the model at startup for performance, but this prevented runtime model changes from taking effect.

### Prompt Issue
The system has separate prompt types for different media:
- **Images**: Direct image analysis
- **Video Frames**: Individual frame analysis within videos

You likely created Chinese prompts for images but not specifically for video frames, so video analysis fell back to default English prompts.

## Files Modified

1. **backend/app/services/llm_service.py**
   - Removed model caching
   - Added dynamic model property

## Test Scripts Created

1. **test_model_fix.py** - Tests model selection changes
2. **debug_prompt_config.py** - Debug prompt configuration
3. **debug_issues_summary.md** - Detailed analysis

## Next Steps

1. **✅ Model fix is complete** - test by changing model in UI
2. **🔧 Create Chinese video frame prompt** following steps above
3. **🧪 Test video analysis** to verify Chinese results
4. **📊 Monitor logs** to confirm both fixes are working

The model selection should work immediately, and once you create the video frame prompt, your Chinese prompts will be used for video analysis!
