"""
German Food Image Scraper
Scrapes images of German dishes from Google Images for fine-tuning BLIP-2
"""

import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
import re
from pathlib import Path
from PIL import Image
from io import BytesIO

class GermanFoodScraper:
    def __init__(self, output_dir="german_food_dataset"):
        self.output_dir = output_dir
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # German dishes with their typical ingredients
        self.german_dishes = {
            # Main Dishes
            'schnitzel': ['pork', 'breadcrumbs', 'lemon', 'butter'],
            'bratwurst': ['sausage', 'mustard', 'bread roll'],
            'sauerbraten': ['beef', 'red cabbage', 'gravy', 'dumplings'],
            'schweinshaxe': ['pork knuckle', 'skin', 'potatoes'],
            'currywurst': ['sausage', 'curry sauce', 'fries'],
            'rouladen': ['beef rolls', 'bacon', 'pickles', 'onions'],
            'kassler': ['smoked pork', 'sauerkraut', 'potatoes'],
            'maultaschen': ['pasta pockets', 'meat', 'spinach'],
            
            # Potato Dishes
            'kartoffelpuffer': ['potato pancakes', 'applesauce', 'sour cream'],
            'bratkartoffeln': ['fried potatoes', 'onions', 'bacon'],
            'kartoffelkloesse': ['potato dumplings', 'gravy'],
            'kartoffelsalat': ['potato salad', 'vinegar', 'onions'],
            
            # Bread & Breakfast
            'brezel': ['pretzel', 'salt', 'butter'],
            'leberkase': ['meatloaf', 'mustard', 'bread'],
            'weisswurst': ['white sausage', 'sweet mustard', 'pretzel'],
            
            # Soups & Stews
            'eintopf': ['stew', 'vegetables', 'meat', 'potatoes'],
            'gulaschsuppe': ['goulash soup', 'beef', 'paprika'],
            'kartoffelsuppe': ['potato soup', 'vegetables', 'sausage'],
            
            # Side Dishes
            'sauerkraut': ['fermented cabbage', 'caraway seeds'],
            'rotkohl': ['red cabbage', 'apples', 'vinegar'],
            'spaetzle': ['egg noodles', 'butter', 'cheese'],
            'knodel': ['dumplings', 'bread', 'parsley'],
            
            # Desserts
            'schwarzwalder kirschtorte': ['black forest cake', 'chocolate', 'cherries', 'cream'],
            'apfelstrudel': ['apple strudel', 'apples', 'cinnamon', 'raisins'],
            'bienenstich': ['bee sting cake', 'almonds', 'custard'],
            'kaiserschmarrn': ['shredded pancake', 'raisins', 'powdered sugar'],
            'lebkuchen': ['gingerbread', 'spices', 'nuts'],
            'stollen': ['christmas bread', 'dried fruits', 'marzipan'],
        }
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)
    
    def scrape_google_images(self, dish_name: str, max_images: int = 10):
        """Scrape images from Google Images."""
        print(f"\n🔍 Searching for: {dish_name}")
        
        # Create dish directory
        dish_dir = Path(self.output_dir) / dish_name.replace(' ', '_')
        dish_dir.mkdir(exist_ok=True)
        
        # Search query
        search_query = f"{dish_name} german food"
        params = {'q': search_query, 'tbm': 'isch', 'hl': 'en', 'safe': 'off'}
        url = f"https://www.google.com/search?{urlencode(params)}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract image URLs using regex (thumbnails)
            pattern = r'https://encrypted-tbn\d\.gstatic\.com/images\?q=tbn:[^"&]*'
            thumbnail_urls = re.findall(pattern, response.text)
            
            # Also try to get img tags
            img_tags = soup.find_all('img')
            for img in img_tags:
                src = img.get('src', '')
                if src.startswith('http') and 'gstatic' in src:
                    thumbnail_urls.append(src)
            
            # Remove duplicates
            thumbnail_urls = list(set(thumbnail_urls))
            
            print(f"   Found {len(thumbnail_urls)} image URLs")
            
            # Download images
            downloaded = 0
            for idx, img_url in enumerate(thumbnail_urls[:max_images * 2]):  # Try more to get enough valid ones
                if downloaded >= max_images:
                    break
                
                try:
                    img_response = requests.get(img_url, headers=self.headers, timeout=10)
                    img = Image.open(BytesIO(img_response.content))
                    
                    # Validate image
                    if img.size[0] < 100 or img.size[1] < 100:
                        continue
                    
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize to reasonable size
                    max_size = 512
                    if img.size[0] > max_size or img.size[1] > max_size:
                        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    
                    # Save image
                    img_path = dish_dir / f"{dish_name.replace(' ', '_')}_{downloaded + 1}.jpg"
                    img.save(img_path, 'JPEG', quality=90)
                    
                    downloaded += 1
                    print(f"   ✅ Downloaded {downloaded}/{max_images}: {img_path.name}")
                    
                except Exception as e:
                    continue
            
            print(f"   ✅ Successfully downloaded {downloaded} images for {dish_name}")
            time.sleep(2)  # Rate limiting
            
            return downloaded
            
        except Exception as e:
            print(f"   ❌ Error scraping {dish_name}: {e}")
            return 0
    
    def scrape_all(self, images_per_dish: int = 10):
        """Scrape images for all German dishes."""
        print("=" * 70)
        print("🇩🇪 German Food Image Scraper")
        print("=" * 70)
        print(f"\nTotal dishes: {len(self.german_dishes)}")
        print(f"Images per dish: {images_per_dish}")
        print(f"Total target images: {len(self.german_dishes) * images_per_dish}")
        print(f"Output directory: {self.output_dir}")
        print("\n" + "=" * 70)
        
        total_downloaded = 0
        for dish_name in self.german_dishes.keys():
            downloaded = self.scrape_google_images(dish_name, images_per_dish)
            total_downloaded += downloaded
        
        print("\n" + "=" * 70)
        print("📊 Scraping Complete!")
        print("=" * 70)
        print(f"Total images downloaded: {total_downloaded}")
        print(f"Total dishes: {len(self.german_dishes)}")
        print(f"Average per dish: {total_downloaded / len(self.german_dishes):.1f}")
        print(f"Dataset location: {self.output_dir}")
        print("=" * 70)

if __name__ == "__main__":
    scraper = GermanFoodScraper()
    scraper.scrape_all(images_per_dish=40)

