/**
 * Indexing notifications component for providing user feedback
 */

import React, { useState, useEffect } from 'react';
import {
  Snackbar,
  Alert,
  AlertTitle,
  Box,
  Typography,
  LinearProgress,
} from '@mui/material';
import { useSearchStore } from '../store/searchStore';

interface IndexingNotification {
  id: string;
  type: 'indexing_started' | 'indexing_completed' | 'indexing_failed' | 'data_refreshed';
  title: string;
  message: string;
  severity: 'success' | 'info' | 'warning' | 'error';
  autoHideDuration?: number;
}

const IndexingNotifications: React.FC = () => {
  const [notifications, setNotifications] = useState<IndexingNotification[]>([]);
  const { isMonitoringIndexing, lastIndexingStatus } = useSearchStore();

  // Track indexing status changes
  useEffect(() => {
    let previousStatus = lastIndexingStatus;

    const handleStatusChange = () => {
      const currentStatus = useSearchStore.getState().lastIndexingStatus;
      
      if (previousStatus !== currentStatus) {
        if (currentStatus === 'processing' && previousStatus !== 'processing') {
          // Indexing started
          addNotification({
            id: `indexing_started_${Date.now()}`,
            type: 'indexing_started',
            title: 'Indexing Started',
            message: 'Media files are being analyzed and indexed...',
            severity: 'info',
            autoHideDuration: 4000,
          });
        } else if (currentStatus === 'completed' && previousStatus === 'processing') {
          // Indexing completed
          addNotification({
            id: `indexing_completed_${Date.now()}`,
            type: 'indexing_completed',
            title: 'Indexing Completed',
            message: 'All media files have been successfully indexed. Your library has been updated!',
            severity: 'success',
            autoHideDuration: 6000,
          });
        } else if (currentStatus === 'failed' && previousStatus === 'processing') {
          // Indexing failed
          addNotification({
            id: `indexing_failed_${Date.now()}`,
            type: 'indexing_failed',
            title: 'Indexing Failed',
            message: 'There was an error during indexing. Please check the logs for details.',
            severity: 'error',
            autoHideDuration: 8000,
          });
        }
        
        previousStatus = currentStatus;
      }
    };

    // Check for status changes periodically while monitoring
    let interval: NodeJS.Timeout;
    if (isMonitoringIndexing) {
      interval = setInterval(handleStatusChange, 1000);
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isMonitoringIndexing, lastIndexingStatus]);

  const addNotification = (notification: IndexingNotification) => {
    setNotifications(prev => [...prev, notification]);
  };

  const removeNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  const handleClose = (id: string) => {
    removeNotification(id);
  };

  return (
    <>
      {notifications.map((notification, index) => (
        <Snackbar
          key={notification.id}
          open={true}
          autoHideDuration={notification.autoHideDuration || 6000}
          onClose={() => handleClose(notification.id)}
          anchorOrigin={{ 
            vertical: 'bottom', 
            horizontal: 'right' 
          }}
          sx={{
            // Stack multiple notifications
            bottom: `${(index * 80) + 24}px !important`,
          }}
        >
          <Alert
            onClose={() => handleClose(notification.id)}
            severity={notification.severity}
            sx={{ 
              width: '100%',
              minWidth: 300,
            }}
          >
            <AlertTitle>{notification.title}</AlertTitle>
            <Typography variant="body2">
              {notification.message}
            </Typography>
            
            {/* Show progress bar for active indexing */}
            {notification.type === 'indexing_started' && isMonitoringIndexing && (
              <Box sx={{ mt: 1 }}>
                <LinearProgress 
                  variant="indeterminate" 
                  sx={{ 
                    height: 4,
                    borderRadius: 2,
                  }} 
                />
              </Box>
            )}
          </Alert>
        </Snackbar>
      ))}
    </>
  );
};

export default IndexingNotifications;
