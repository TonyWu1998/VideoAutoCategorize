/**
 * Comprehensive test to verify search store functionality including indexing synchronization
 */

import { useSearchStore } from './store/searchStore';

// Test the new methods
const testSearchStore = () => {
  const store = useSearchStore.getState();

  console.log('=== Testing Search Store Functionality ===');

  // Test clearAllData
  console.log('\n1. Testing clearAllData...');
  store.clearAllData();
  const stateAfterClear = useSearchStore.getState();
  console.log('State after clearAllData:', {
    allMediaItems: stateAfterClear.allMediaItems.length,
    searchResults: stateAfterClear.searchResults.length,
    totalResults: stateAfterClear.totalResults,
    query: stateAfterClear.query,
    selectedItems: stateAfterClear.selectedItems.size,
    isMonitoringIndexing: stateAfterClear.isMonitoringIndexing,
    lastIndexingStatus: stateAfterClear.lastIndexingStatus,
  });

  // Test removeItemFromState with mock data
  console.log('\n2. Testing removeItemFromState...');

  // First, add some mock data
  const mockItems = [
    { file_id: 'test1', metadata: { file_name: 'test1.mp4' } },
    { file_id: 'test2', metadata: { file_name: 'test2.mp4' } },
    { file_id: 'test3', metadata: { file_name: 'test3.mp4' } },
  ] as any[];

  // Manually set state for testing
  useSearchStore.setState({
    allMediaItems: mockItems,
    searchResults: mockItems,
    totalResults: mockItems.length,
    selectedItems: new Set(['test1', 'test2']),
  });

  console.log('State before removal:', {
    allMediaItems: useSearchStore.getState().allMediaItems.length,
    searchResults: useSearchStore.getState().searchResults.length,
    selectedItems: Array.from(useSearchStore.getState().selectedItems),
  });

  // Remove one item
  store.removeItemFromState('test2');

  const stateAfterRemoval = useSearchStore.getState();
  console.log('State after removing test2:', {
    allMediaItems: stateAfterRemoval.allMediaItems.length,
    searchResults: stateAfterRemoval.searchResults.length,
    selectedItems: Array.from(stateAfterRemoval.selectedItems),
    remainingItems: stateAfterRemoval.allMediaItems.map(item => item.file_id),
  });

  // Test indexing monitoring
  console.log('\n3. Testing indexing monitoring...');
  console.log('Initial monitoring state:', {
    isMonitoringIndexing: useSearchStore.getState().isMonitoringIndexing,
    lastIndexingStatus: useSearchStore.getState().lastIndexingStatus,
  });

  // Test starting monitor
  console.log('Starting indexing monitor...');
  store.startIndexingMonitor();

  setTimeout(() => {
    const monitoringState = useSearchStore.getState();
    console.log('Monitoring state after start:', {
      isMonitoringIndexing: monitoringState.isMonitoringIndexing,
      lastIndexingStatus: monitoringState.lastIndexingStatus,
    });

    // Test stopping monitor
    console.log('Stopping indexing monitor...');
    store.stopIndexingMonitor();

    setTimeout(() => {
      const finalState = useSearchStore.getState();
      console.log('Final monitoring state:', {
        isMonitoringIndexing: finalState.isMonitoringIndexing,
        lastIndexingStatus: finalState.lastIndexingStatus,
      });

      console.log('\n=== Search Store Tests Completed! ===');
    }, 1000);
  }, 2000);
};

// Export for use in browser console
(window as any).testSearchStore = testSearchStore;

export default testSearchStore;
