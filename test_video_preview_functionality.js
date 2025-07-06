/**
 * Test script for video preview functionality
 * Tests the new video thumbnail, multiple frames, and file explorer features
 */

const puppeteer = require('puppeteer');

async function testVideoPreviewFunctionality() {
    console.log('🎬 Testing Video Preview Functionality...');
    
    const browser = await puppeteer.launch({ 
        headless: false,
        defaultViewport: { width: 1280, height: 720 }
    });
    
    try {
        const page = await browser.newPage();
        
        // Navigate to the application
        console.log('📱 Navigating to application...');
        await page.goto('http://127.0.0.1:5174/', { waitUntil: 'networkidle0' });
        
        // Wait for the search interface to load
        await page.waitForSelector('input[placeholder*="search"]', { timeout: 10000 });
        console.log('✅ Application loaded successfully');
        
        // Perform a search to find video content
        console.log('🔍 Searching for video content...');
        const searchInput = await page.$('input[placeholder*="search"]');
        await searchInput.type('chinese');
        
        // Click search button
        const searchButton = await page.$('button[type="submit"], button:has-text("Search")');
        if (searchButton) {
            await searchButton.click();
        } else {
            // Try pressing Enter
            await searchInput.press('Enter');
        }
        
        // Wait for search results
        await page.waitForTimeout(3000);
        console.log('⏳ Waiting for search results...');
        
        // Look for video results (cards with play icon)
        const videoCards = await page.$$('[data-testid="media-card"], .MuiCard-root');
        console.log(`📹 Found ${videoCards.length} media cards`);
        
        if (videoCards.length > 0) {
            // Test 1: Check if video thumbnails are displayed
            console.log('🖼️ Testing video thumbnail display...');
            const firstCard = videoCards[0];
            
            // Check if thumbnail image is present
            const thumbnail = await firstCard.$('img');
            if (thumbnail) {
                const src = await thumbnail.evaluate(img => img.src);
                console.log(`✅ Thumbnail found: ${src}`);
                
                // Check if it's a video (should have play icon overlay)
                const playIcon = await firstCard.$('[data-testid="play-icon"], svg[data-testid="PlayArrowIcon"]');
                if (playIcon) {
                    console.log('✅ Video play icon overlay found');
                } else {
                    console.log('⚠️ Video play icon overlay not found');
                }
            } else {
                console.log('❌ No thumbnail image found');
            }
            
            // Test 2: Click on video to open MediaViewer
            console.log('🎥 Testing MediaViewer with video frames...');
            await firstCard.click();
            
            // Wait for modal to open
            await page.waitForSelector('[role="dialog"], .MuiDialog-root', { timeout: 5000 });
            console.log('✅ MediaViewer modal opened');
            
            // Wait for video frames to load
            await page.waitForTimeout(3000);
            
            // Check for video frames preview section
            const framesSection = await page.$('h6:has-text("Video Preview Frames"), [data-testid="video-frames"]');
            if (framesSection) {
                console.log('✅ Video frames preview section found');
                
                // Count the number of frame images
                const frameImages = await page.$$('img[alt*="Frame"], [data-testid="video-frame"]');
                console.log(`📸 Found ${frameImages.length} video frame previews`);
                
                if (frameImages.length >= 3) {
                    console.log('✅ Multiple video frames displayed successfully');
                } else {
                    console.log('⚠️ Expected at least 3 video frames');
                }
            } else {
                console.log('❌ Video frames preview section not found');
            }
            
            // Test 3: Check for "Open in File Explorer" button
            console.log('📁 Testing "Open in File Explorer" functionality...');
            const explorerButton = await page.$('button:has-text("Open in Explorer"), [data-testid="open-explorer-btn"]');
            if (explorerButton) {
                console.log('✅ "Open in File Explorer" button found');
                
                // Click the button to test functionality
                await explorerButton.click();
                
                // Wait for potential success message
                await page.waitForTimeout(2000);
                
                // Check for success notification
                const successMessage = await page.$('.MuiAlert-root, [role="alert"]');
                if (successMessage) {
                    const messageText = await successMessage.evaluate(el => el.textContent);
                    console.log(`✅ Explorer action result: ${messageText}`);
                } else {
                    console.log('⚠️ No notification message found after clicking explorer button');
                }
            } else {
                console.log('❌ "Open in File Explorer" button not found');
            }
            
            // Test 4: Check other action buttons
            console.log('🔧 Testing other action buttons...');
            const downloadBtn = await page.$('button:has-text("Download")');
            const similarBtn = await page.$('button:has-text("Find Similar")');
            
            if (downloadBtn) console.log('✅ Download button found');
            if (similarBtn) console.log('✅ Find Similar button found');
            
            // Close the modal
            const closeButton = await page.$('[aria-label="close"], button:has-text("×")');
            if (closeButton) {
                await closeButton.click();
                console.log('✅ Modal closed successfully');
            }
            
        } else {
            console.log('❌ No video cards found in search results');
        }
        
        console.log('🎉 Video preview functionality testing completed!');
        
    } catch (error) {
        console.error('❌ Test failed:', error);
    } finally {
        await browser.close();
    }
}

// Run the test
testVideoPreviewFunctionality().catch(console.error);
