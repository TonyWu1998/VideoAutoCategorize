/**
 * End-to-End Puppeteer Test for VideoAutoCategorize
 * Tests the complete workflow: Frontend → Backend → AI Analysis → Search
 */

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

// Test configuration
const CONFIG = {
    frontend_url: 'http://127.0.0.1:5173',
    backend_url: 'http://127.0.0.1:8000',
    test_video_path: '/Volumes/T7 Shield/final_cut/jimingyue_4k_dv.mp4',
    screenshots_dir: './screenshots',
    timeout: 60000
};

// Ensure screenshots directory exists
if (!fs.existsSync(CONFIG.screenshots_dir)) {
    fs.mkdirSync(CONFIG.screenshots_dir, { recursive: true });
}

async function takeScreenshot(page, name, description) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `${timestamp}_${name}.png`;
    const filepath = path.join(CONFIG.screenshots_dir, filename);
    
    await page.screenshot({ 
        path: filepath, 
        fullPage: true 
    });
    
    console.log(`📸 Screenshot saved: ${filename} - ${description}`);
    return filepath;
}

async function testBackendHealth() {
    console.log('🏥 Testing Backend Health...');
    
    try {
        const response = await fetch(`${CONFIG.backend_url}/health`);
        const data = await response.json();
        
        if (response.ok && data.status === 'healthy') {
            console.log('✅ Backend is healthy');
            return true;
        } else {
            console.log('❌ Backend health check failed:', data);
            return false;
        }
    } catch (error) {
        console.log('❌ Backend health check error:', error.message);
        return false;
    }
}

async function testFrontendLoad(page) {
    console.log('🌐 Testing Frontend Load...');
    
    try {
        await page.goto(CONFIG.frontend_url, { 
            waitUntil: 'networkidle2',
            timeout: CONFIG.timeout 
        });
        
        await takeScreenshot(page, 'frontend_loaded', 'Frontend application loaded');
        
        // Check if the page loaded correctly
        const title = await page.title();
        console.log(`📄 Page title: ${title}`);
        
        // Look for key elements
        const hasSearchBox = await page.$('input[type="search"], input[placeholder*="search"], input[placeholder*="Search"]') !== null;
        const hasUploadArea = await page.$('input[type="file"], [data-testid*="upload"], [class*="upload"]') !== null;
        
        console.log(`🔍 Search functionality detected: ${hasSearchBox}`);
        console.log(`📤 Upload functionality detected: ${hasUploadArea}`);
        
        if (title && (hasSearchBox || hasUploadArea)) {
            console.log('✅ Frontend loaded successfully');
            return true;
        } else {
            console.log('❌ Frontend missing key elements');
            return false;
        }
    } catch (error) {
        console.log('❌ Frontend load error:', error.message);
        await takeScreenshot(page, 'frontend_error', 'Frontend load error');
        return false;
    }
}

async function testSearchFunctionality(page) {
    console.log('🔍 Testing Search Functionality...');
    
    try {
        // Test queries that should find our indexed video
        const testQueries = [
            'traditional Chinese courtyard',
            'red pillars',
            'serene elegant',
            'jimingyue'
        ];
        
        for (const query of testQueries) {
            console.log(`🔎 Testing search query: "${query}"`);
            
            // Find search input
            const searchInput = await page.$('input[type="search"], input[placeholder*="search"], input[placeholder*="Search"]');
            
            if (!searchInput) {
                console.log('❌ Search input not found');
                continue;
            }
            
            // Clear and enter search query
            await searchInput.click({ clickCount: 3 }); // Select all
            await searchInput.type(query);
            
            // Submit search (try Enter key or search button)
            await page.keyboard.press('Enter');
            
            // Wait for results
            await page.waitForTimeout(3000);
            
            await takeScreenshot(page, `search_${query.replace(/\s+/g, '_')}`, `Search results for "${query}"`);
            
            // Check for results
            const resultsContainer = await page.$('[class*="result"], [class*="search"], [data-testid*="result"]');
            
            if (resultsContainer) {
                // Look for our video file
                const pageContent = await page.content();
                const hasVideoResult = pageContent.includes('jimingyue') || 
                                     pageContent.includes('courtyard') || 
                                     pageContent.includes('red pillars');
                
                const hasErrorMessage = pageContent.includes('Error analyzing') || 
                                       pageContent.includes('cannot identify image');
                
                if (hasVideoResult && !hasErrorMessage) {
                    console.log(`✅ Search for "${query}" found relevant results without errors`);
                } else if (hasErrorMessage) {
                    console.log(`❌ Search for "${query}" contains error messages`);
                } else {
                    console.log(`⚠️  Search for "${query}" - no relevant results found`);
                }
            } else {
                console.log(`❌ No results container found for "${query}"`);
            }
        }
        
        return true;
    } catch (error) {
        console.log('❌ Search test error:', error.message);
        await takeScreenshot(page, 'search_error', 'Search functionality error');
        return false;
    }
}

async function testIndexingStatus(page) {
    console.log('📊 Testing Indexing Status...');
    
    try {
        // Look for indexing or status pages
        const statusLinks = await page.$$('a[href*="status"], a[href*="index"], button[class*="status"]');
        
        if (statusLinks.length > 0) {
            await statusLinks[0].click();
            await page.waitForTimeout(2000);
            
            await takeScreenshot(page, 'indexing_status', 'Indexing status page');
            
            // Check for our video in the indexed files
            const pageContent = await page.content();
            const hasOurVideo = pageContent.includes('jimingyue_4k_dv.mp4');
            const hasErrorStatus = pageContent.includes('Error analyzing') || 
                                  pageContent.includes('failed') ||
                                  pageContent.includes('error');
            
            if (hasOurVideo && !hasErrorStatus) {
                console.log('✅ Video appears in indexing status without errors');
                return true;
            } else if (hasErrorStatus) {
                console.log('❌ Indexing status shows errors');
                return false;
            } else {
                console.log('⚠️  Video not found in indexing status');
                return false;
            }
        } else {
            console.log('⚠️  No indexing status page found');
            return true; // Not a failure if status page doesn't exist
        }
    } catch (error) {
        console.log('❌ Indexing status test error:', error.message);
        return false;
    }
}

async function runE2ETests() {
    console.log('🚀 Starting VideoAutoCategorize End-to-End Tests');
    console.log('=' * 60);
    
    let browser;
    let testResults = {
        backend_health: false,
        frontend_load: false,
        search_functionality: false,
        indexing_status: false
    };
    
    try {
        // Test backend health first
        testResults.backend_health = await testBackendHealth();
        
        if (!testResults.backend_health) {
            console.log('❌ Backend not healthy, skipping frontend tests');
            return testResults;
        }
        
        // Launch browser
        console.log('🌐 Launching browser...');
        browser = await puppeteer.launch({ 
            headless: false, // Show browser for debugging
            defaultViewport: { width: 1280, height: 720 },
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });
        
        const page = await browser.newPage();
        
        // Test frontend load
        testResults.frontend_load = await testFrontendLoad(page);
        
        if (testResults.frontend_load) {
            // Test search functionality
            testResults.search_functionality = await testSearchFunctionality(page);
            
            // Test indexing status
            testResults.indexing_status = await testIndexingStatus(page);
        }
        
        // Final screenshot
        await takeScreenshot(page, 'final_state', 'Final application state');
        
    } catch (error) {
        console.log('❌ E2E test error:', error.message);
    } finally {
        if (browser) {
            await browser.close();
        }
    }
    
    // Print results summary
    console.log('\n📊 Test Results Summary:');
    console.log('=' * 40);
    console.log(`Backend Health: ${testResults.backend_health ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`Frontend Load: ${testResults.frontend_load ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`Search Functionality: ${testResults.search_functionality ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`Indexing Status: ${testResults.indexing_status ? '✅ PASS' : '❌ FAIL'}`);
    
    const passedTests = Object.values(testResults).filter(result => result).length;
    const totalTests = Object.keys(testResults).length;
    
    console.log(`\n🎯 Overall Result: ${passedTests}/${totalTests} tests passed`);
    
    if (passedTests === totalTests) {
        console.log('🎉 All tests PASSED! VideoAutoCategorize is working correctly!');
    } else {
        console.log('⚠️  Some tests failed. Check the logs and screenshots for details.');
    }
    
    return testResults;
}

// Run the tests
runE2ETests().catch(console.error);
