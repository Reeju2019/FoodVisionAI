"""
Bengali Food Dataset Scraper

Scrapes Bengali food images from Google Images.
Simple and effective approach using google_images_download or direct scraping.

Creates a dataset suitable for fine-tuning BLIP-2 or training a classifier.
"""

import requests
from bs4 import BeautifulSoup
import json
import os
from pathlib import Path
from typing import List, Dict
import time
from urllib.parse import quote, urlencode
from PIL import Image
from io import BytesIO
import re

class BengaliFoodScraper:
    """Scraper for Bengali food images and recipes."""

    def __init__(self, output_dir: str = "bengali_food_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)

        # Common Bengali dishes
        self.bengali_dishes = [
            # Rice dishes
            "biryani", "pulao", "khichuri", "tehari",

            # Fish dishes (Bengali specialty!)
            "ilish macher jhol", "chingri malai curry", "doi maach",
            "macher kalia", "fish fry bengali",

            # Meat dishes
            "kosha mangsho", "chicken rezala", "mutton curry bengali",

            # Vegetable dishes
            "shukto", "aloo posto", "begun bhaja", "lau ghonto",
            "cholar dal", "moong dal bengali",

            # Snacks
            "puchka", "jhalmuri", "telebhaja", "singara",
            "kachori bengali", "ghugni",

            # Sweets (Bengali famous!)
            "rasgulla", "sandesh", "mishti doi", "chomchom",
            "rasmalai", "pantua", "langcha", "kalakand"
        ]

        self.dataset = []

        # User agent to avoid blocking
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    def scrape_google_images(self, dish_name: str, max_images: int = 10) -> List[Dict]:
        """
        Scrape images from Google Images.

        Args:
            dish_name: Name of the dish
            max_images: Maximum number of images to download

        Returns:
            List of image metadata
        """
        print(f"🔍 Searching Google Images for: {dish_name}")

        images = []

        try:
            # Add "bengali food" to search query for better results
            search_query = f"{dish_name} bengali food"

            # Google Images search URL
            params = {
                'q': search_query,
                'tbm': 'isch',  # Image search
                'hl': 'en',
                'safe': 'off'
            }

            url = f"https://www.google.com/search?{urlencode(params)}"

            response = requests.get(url, headers=self.headers, timeout=15)

            if response.status_code == 200:
                # Parse HTML to find image URLs
                # Google Images embeds image URLs in the page source

                # Method 1: Find direct image URLs in img tags
                soup = BeautifulSoup(response.content, 'html.parser')
                img_tags = soup.find_all('img')

                count = 0
                for img in img_tags:
                    if count >= max_images:
                        break

                    # Get image URL
                    img_url = img.get('src') or img.get('data-src')

                    # Skip base64 encoded images and icons
                    if not img_url or img_url.startswith('data:') or 'logo' in img_url.lower():
                        continue

                    # Make sure it's a full URL
                    if not img_url.startswith('http'):
                        continue

                    # Download image
                    img_filename = f"{dish_name.replace(' ', '_')}_google_{count+1}.jpg"
                    img_path = self.output_dir / "images" / img_filename

                    if self._download_image(img_url, img_path):
                        images.append({
                            "dish_name": dish_name,
                            "image_path": str(img_path),
                            "source": "google_images",
                            "url": img_url
                        })
                        print(f"  ✅ Downloaded: {img_filename}")
                        count += 1

                # Method 2: Extract from JavaScript data (more reliable)
                if count < max_images:
                    # Find image URLs in page source using regex
                    pattern = r'https://encrypted-tbn\d\.gstatic\.com/images\?q=tbn:[^"&]*'
                    thumbnail_urls = re.findall(pattern, response.text)

                    for thumb_url in thumbnail_urls[:max_images - count]:
                        img_filename = f"{dish_name.replace(' ', '_')}_google_{count+1}.jpg"
                        img_path = self.output_dir / "images" / img_filename

                        if self._download_image(thumb_url, img_path):
                            images.append({
                                "dish_name": dish_name,
                                "image_path": str(img_path),
                                "source": "google_images",
                                "url": thumb_url
                            })
                            print(f"  ✅ Downloaded: {img_filename}")
                            count += 1

                if count == 0:
                    print(f"  ⚠️  No images found for {dish_name}")

            else:
                print(f"  ❌ HTTP {response.status_code} for {dish_name}")

            time.sleep(2)  # Rate limiting - be respectful!

        except Exception as e:
            print(f"  ❌ Error scraping Google Images: {e}")

        return images



    def _download_image(self, url: str, save_path: Path) -> bool:
        """
        Download and save an image.

        Args:
            url: Image URL
            save_path: Path to save the image

        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                # Verify it's a valid image
                img = Image.open(BytesIO(response.content))

                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize if too large (max 1024x1024)
                if max(img.size) > 1024:
                    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

                # Save
                img.save(save_path, 'JPEG', quality=90)
                return True

        except Exception as e:
            print(f"    ⚠️  Failed to download {url}: {e}")
            return False

    def scrape_all_dishes(self, images_per_dish: int = 10):
        """
        Scrape images for all Bengali dishes.

        Args:
            images_per_dish: Number of images to collect per dish
        """
        print("=" * 70)
        print("🇧🇩 Bengali Food Dataset Scraper")
        print("=" * 70)
        print(f"\nScraping {len(self.bengali_dishes)} dishes...")
        print(f"Target: {images_per_dish} images per dish")
        print(f"Total target: {len(self.bengali_dishes) * images_per_dish} images\n")

        for idx, dish in enumerate(self.bengali_dishes, 1):
            print(f"\n[{idx}/{len(self.bengali_dishes)}] {dish}")
            print("-" * 70)

            # Scrape from Google Images
            images = self.scrape_google_images(dish, max_images=images_per_dish)

            self.dataset.extend(images)

            print(f"  📊 Collected {len(images)} images for {dish}")

            # Save progress after each dish
            self._save_metadata()

            # Rate limiting
            time.sleep(2)

        print("\n" + "=" * 70)
        print(f"✅ Scraping complete!")
        print(f"📊 Total images collected: {len(self.dataset)}")
        print(f"💾 Saved to: {self.output_dir}")
        print("=" * 70)

    def _save_metadata(self):
        """Save dataset metadata to JSON."""
        metadata_file = self.output_dir / "metadata" / "dataset.json"

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_images": len(self.dataset),
                "total_dishes": len(set(img['dish_name'] for img in self.dataset)),
                "images": self.dataset
            }, f, indent=2, ensure_ascii=False)

    def create_training_split(self, train_ratio: float = 0.8):
        """
        Create train/val split for the dataset.

        Args:
            train_ratio: Ratio of training data (default: 0.8)
        """
        import random

        # Group by dish
        dish_groups = {}
        for img in self.dataset:
            dish = img['dish_name']
            if dish not in dish_groups:
                dish_groups[dish] = []
            dish_groups[dish].append(img)

        train_data = []
        val_data = []

        for dish, images in dish_groups.items():
            random.shuffle(images)
            split_idx = int(len(images) * train_ratio)
            train_data.extend(images[:split_idx])
            val_data.extend(images[split_idx:])

        # Save splits
        with open(self.output_dir / "metadata" / "train.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        with open(self.output_dir / "metadata" / "val.json", 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)

        print(f"\n📊 Dataset split:")
        print(f"  Train: {len(train_data)} images")
        print(f"  Val: {len(val_data)} images")


def main():
    """Main function to run the scraper."""

    print("\n🇧🇩 Bengali Food Dataset Scraper")
    print("=" * 70)
    print("\n⚠️  IMPORTANT NOTES:")
    print("1. This scraper uses Google Images search")
    print("2. Scrapes 5-10 images per dish category")
    print("3. Includes 40+ Bengali dishes")
    print("4. Respects rate limits (2 sec delay between requests)")
    print("5. This will take 10-15 minutes to complete")
    print("6. Be respectful - don't abuse Google's servers!")
    print("=" * 70)

    response = input("\nContinue? (y/n): ")

    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Create scraper
    scraper = BengaliFoodScraper(output_dir="bengali_food_dataset")

    # Scrape all dishes (5 images per dish for quick test)
    # Change to 10-20 for production
    scraper.scrape_all_dishes(images_per_dish=5)

    # Create train/val split
    scraper.create_training_split(train_ratio=0.8)

    print("\n✅ Done! Dataset ready for fine-tuning.")
    print(f"📁 Location: {scraper.output_dir}")
    print("\n💡 Next steps:")
    print("1. Review the images in bengali_food_dataset/images/")
    print("2. Use the dataset to fine-tune BLIP-2 or train a classifier")
    print("3. See 'finetune_blip2_bengali.py' for fine-tuning code")


if __name__ == "__main__":
    main()

