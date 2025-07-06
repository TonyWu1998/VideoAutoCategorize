#!/usr/bin/env python3
"""
Create test media files for demonstrating AI analysis capabilities.
"""

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

def create_test_images():
    """Create diverse test images for AI analysis."""
    
    # Create test_media directory
    os.makedirs("test_media/images", exist_ok=True)
    
    # Test Image 1: Sunset landscape
    img1 = Image.new('RGB', (800, 600), color='skyblue')
    draw1 = ImageDraw.Draw(img1)
    
    # Draw sunset scene
    # Sky gradient
    for y in range(300):
        color_intensity = int(255 * (1 - y/300))
        draw1.rectangle([0, y, 800, y+1], fill=(255, color_intensity, color_intensity//2))
    
    # Sun
    draw1.ellipse([350, 100, 450, 200], fill='yellow', outline='orange', width=3)
    
    # Mountains
    draw1.polygon([(0, 400), (200, 250), (400, 300), (600, 200), (800, 350), (800, 600), (0, 600)], 
                  fill='darkgreen')
    
    # Water reflection
    for y in range(400, 600):
        alpha = (y - 400) / 200
        color = (int(135 * alpha), int(206 * alpha), int(235 * alpha))
        draw1.rectangle([0, y, 800, y+1], fill=color)
    
    img1.save("test_media/images/sunset_landscape.jpg", "JPEG", quality=95)
    print("✅ Created: sunset_landscape.jpg")
    
    # Test Image 2: City skyline
    img2 = Image.new('RGB', (800, 600), color='lightblue')
    draw2 = ImageDraw.Draw(img2)
    
    # Buildings
    buildings = [
        (50, 200, 150, 600),   # Building 1
        (180, 150, 250, 600),  # Building 2
        (280, 250, 350, 600),  # Building 3
        (380, 100, 450, 600),  # Building 4 (tallest)
        (480, 180, 550, 600),  # Building 5
        (580, 220, 650, 600),  # Building 6
        (680, 160, 750, 600),  # Building 7
    ]
    
    colors = ['gray', 'darkgray', 'lightgray', 'dimgray', 'silver', 'gainsboro', 'darkslategray']
    
    for i, (x1, y1, x2, y2) in enumerate(buildings):
        draw2.rectangle([x1, y1, x2, y2], fill=colors[i % len(colors)], outline='black')
        
        # Windows
        for window_y in range(y1 + 20, y2 - 20, 40):
            for window_x in range(x1 + 10, x2 - 10, 25):
                if np.random.random() > 0.3:  # Some windows are lit
                    draw2.rectangle([window_x, window_y, window_x + 15, window_y + 20], 
                                  fill='yellow' if np.random.random() > 0.5 else 'white')
    
    img2.save("test_media/images/city_skyline.jpg", "JPEG", quality=95)
    print("✅ Created: city_skyline.jpg")
    
    # Test Image 3: Nature scene with animals
    img3 = Image.new('RGB', (800, 600), color='lightgreen')
    draw3 = ImageDraw.Draw(img3)
    
    # Forest background
    draw3.rectangle([0, 0, 800, 400], fill='forestgreen')
    draw3.rectangle([0, 400, 800, 600], fill='green')
    
    # Trees
    tree_positions = [(100, 200), (300, 180), (500, 220), (700, 190)]
    for x, y in tree_positions:
        # Trunk
        draw3.rectangle([x-10, y, x+10, y+200], fill='brown')
        # Leaves
        draw3.ellipse([x-40, y-50, x+40, y+50], fill='darkgreen')
    
    # Animals (simple shapes)
    # Deer
    draw3.ellipse([200, 450, 250, 480], fill='brown')  # Body
    draw3.ellipse([240, 440, 260, 460], fill='brown')  # Head
    draw3.line([210, 450, 210, 500], fill='brown', width=3)  # Legs
    draw3.line([240, 450, 240, 500], fill='brown', width=3)
    
    # Bird
    draw3.ellipse([400, 300, 420, 315], fill='red')
    draw3.polygon([(420, 307), (435, 305), (430, 312)], fill='orange')  # Beak
    
    img3.save("test_media/images/forest_animals.jpg", "JPEG", quality=95)
    print("✅ Created: forest_animals.jpg")
    
    # Test Image 4: Beach scene
    img4 = Image.new('RGB', (800, 600), color='skyblue')
    draw4 = ImageDraw.Draw(img4)
    
    # Sky
    draw4.rectangle([0, 0, 800, 300], fill='lightblue')
    
    # Ocean
    for y in range(300, 450):
        blue_intensity = int(100 + (y - 300) * 0.5)
        draw4.rectangle([0, y, 800, y+1], fill=(0, 0, blue_intensity))
    
    # Beach
    draw4.rectangle([0, 450, 800, 600], fill='sandybrown')
    
    # Palm tree
    draw4.rectangle([100, 350, 120, 500], fill='brown')  # Trunk
    # Palm fronds
    frond_points = [
        [(120, 350), (180, 320), (200, 340), (140, 360)],
        [(120, 350), (160, 280), (180, 300), (130, 350)],
        [(120, 350), (80, 320), (60, 340), (110, 360)],
        [(120, 350), (90, 280), (70, 300), (110, 350)]
    ]
    for points in frond_points:
        draw4.polygon(points, fill='green')
    
    # Sun
    draw4.ellipse([650, 50, 750, 150], fill='yellow')
    
    # Waves
    for x in range(0, 800, 50):
        draw4.arc([x, 420, x+50, 450], 0, 180, fill='white', width=2)
    
    img4.save("test_media/images/beach_scene.jpg", "JPEG", quality=95)
    print("✅ Created: beach_scene.jpg")

def create_test_video():
    """Create a simple test video."""
    
    os.makedirs("test_media/videos", exist_ok=True)
    
    # Create a simple animated video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_media/videos/moving_circle.mp4', fourcc, 10.0, (640, 480))
    
    for frame in range(50):  # 5 seconds at 10 fps
        # Create frame
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (50, 100, 150)  # Background color
        
        # Moving circle
        x = int(50 + (frame / 50) * 540)  # Move from left to right
        y = 240  # Center vertically
        cv2.circle(img, (x, y), 30, (0, 255, 255), -1)  # Yellow circle
        
        # Add frame number text
        cv2.putText(img, f'Frame {frame+1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(img)
    
    out.release()
    print("✅ Created: moving_circle.mp4")

if __name__ == "__main__":
    print("🎨 Creating test media files...")
    create_test_images()
    create_test_video()
    print("\n✅ Test media creation completed!")
    print("\nCreated files:")
    print("📁 test_media/images/")
    print("  - sunset_landscape.jpg (landscape with sunset)")
    print("  - city_skyline.jpg (urban scene with buildings)")
    print("  - forest_animals.jpg (nature scene with wildlife)")
    print("  - beach_scene.jpg (tropical beach with palm tree)")
    print("📁 test_media/videos/")
    print("  - moving_circle.mp4 (animated circle video)")
