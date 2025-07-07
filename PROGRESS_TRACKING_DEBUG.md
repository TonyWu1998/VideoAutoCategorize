# Progress Tracking Debug Guide

## Issues Identified and Fixed

### 1. Backend Progress Tracking Issues

**Problem**: The progress tracking in both regular indexing and reindexing was inconsistent:
- In reindexing: `processed_files` was never incremented
- In regular indexing: `processed_files` was incremented by batch size regardless of actual processing results

**Fix Applied**:
- Added `processed_files` increment in reindexing for each file (success, failure, or skip)
- Modified regular indexing to increment `processed_files` per individual file rather than per batch
- Ensured consistent progress tracking across all indexing operations

### 2. UI Refresh Trigger Issues

**Problem**: The UI refresh logic only triggered on status changes from 'processing' to 'completed', but didn't handle cases where:
- Status remained 'processing' but progress was actually complete
- Progress tracking was inconsistent

**Fix Applied**:
- Enhanced monitoring logic to also check for progress completion
- Added fallback trigger when `processed_files >= total_files`
- Improved logging for better debugging

### 3. Frontend Progress Display Issues

**Problem**: Limited debugging information made it difficult to identify progress tracking problems

**Fix Applied**:
- Added detailed console logging in progress calculation
- Enhanced error handling and status reporting
- Added manual refresh button for debugging and user control

## Debugging Steps

### 1. Check Backend Progress Tracking

```bash
# Check backend logs for progress updates
tail -f backend/logs/app.log | grep -E "(progress|processed|total)"
```

### 2. Check Frontend Console

Open browser developer tools and look for:
- Progress calculation logs
- Indexing monitor status changes
- Refresh trigger events

### 3. Manual Testing

1. Start an indexing operation
2. Monitor the Task Progress Dashboard
3. Check that progress increments properly (not stuck at 0/X)
4. Verify UI refreshes when indexing completes
5. Use manual refresh button if needed

### 4. API Status Check

```javascript
// In browser console
fetch('/api/index/status')
  .then(r => r.json())
  .then(data => console.log('Current status:', data));
```

## Expected Behavior After Fixes

### Progress Tracking
- `total_files`: Set correctly during scanning phase
- `processed_files`: Increments for each file processed (success or failure)
- `successful_files`: Increments only for successfully processed files
- `failed_files`: Increments only for failed files

### UI Updates
- Progress bar shows accurate percentage based on `processed_files / total_files`
- Frame-level progress shows for video analysis
- UI automatically refreshes when indexing completes
- Manual refresh button available for debugging

### Monitoring Logic
- Starts automatically when indexing begins
- Polls every 2 seconds during active indexing
- Triggers refresh on status change OR progress completion
- Stops monitoring when indexing is done

## Common Issues and Solutions

### Issue: Progress Stuck at 0/X
**Cause**: Backend not incrementing `processed_files`
**Solution**: Check backend logs, restart indexing service if needed

### Issue: UI Not Refreshing After Completion
**Cause**: Monitoring not detecting completion
**Solution**: Use manual refresh button, check console for monitoring logs

### Issue: Inconsistent Progress Display
**Cause**: Race condition between status updates
**Solution**: Enhanced monitoring logic should handle this automatically

### Issue: Status Shows 'Processing' but No Progress
**Cause**: Indexing job may be stuck or crashed
**Solution**: Cancel job and restart, check backend logs for errors

## Testing Checklist

- [ ] Start indexing operation
- [ ] Verify progress increments from 0 to total
- [ ] Check that successful/failed counts are accurate
- [ ] Confirm UI refreshes automatically when complete
- [ ] Test manual refresh button
- [ ] Verify new content appears in main view
- [ ] Test reindexing operations
- [ ] Check batch reindexing progress

## Monitoring Commands

```javascript
// Monitor indexing status in browser console
const monitorStatus = () => {
  setInterval(async () => {
    const response = await fetch('/api/index/status');
    const status = await response.json();
    console.log('Status:', status.status, 
                'Progress:', status.progress?.processed_files + '/' + status.progress?.total_files,
                'Success:', status.progress?.successful_files,
                'Failed:', status.progress?.failed_files);
  }, 2000);
};

// Start monitoring
monitorStatus();
```

## Performance Considerations

- Progress updates are logged at debug level to avoid log spam
- Monitoring polls every 2 seconds (reasonable balance)
- UI updates are batched to prevent excessive re-renders
- Manual refresh provides escape hatch for edge cases
