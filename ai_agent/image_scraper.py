import asyncio
import aiohttp
from bs4 import BeautifulSoup, Tag
from PIL import Image
from io import BytesIO
from typing import List, Dict, Optional, Tuple, Union
from utils import logger
import random
import json
from urllib.parse import urljoin, quote_plus
import hashlib
import os
from fake_useragent import UserAgent
import re
from image_tools import verify_image, optimize_image
from collections import Counter

# Optional NLP support
try:
    import spacy
    SPACY_AVAILABLE = True
    nlp = spacy.load('en_core_web_sm')
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None

# Optional CLIP support
CLIP_AVAILABLE = False
try:
    import torch
    from PIL import Image
    import clip
    CLIP_AVAILABLE = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
except (ImportError, ModuleNotFoundError):
    logger.warning("CLIP not available. Some image processing features will be limited.")

class APIConfig:
    PIXABAY_API_KEY = os.getenv('PIXABAY_API_KEY', '')
    PEXELS_API_KEY = os.getenv('PEXELS_API_KEY', '')
    UNSPLASH_ACCESS_KEY = os.getenv('UNSPLASH_ACCESS_KEY', '')

class ImageScraper:
    def __init__(self):
        self.ua = UserAgent()
        self.session = None
        self.headers = {'User-Agent': self.ua.random}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_images(self, topic: str, num_images: int = 3) -> Optional[Dict[str, Union[str, List[str]]]]:
        # This method is required by the system and should call get_images internally
        return await self.get_images(topic, num_images)

    async def get_images(self, topic: str, num_images: int = 3) -> Optional[Dict[str, Union[str, List[str]]]]:
        try:
            # Try API-based image fetching first
            image_results = await self._get_images_from_apis(topic, num_images)
            if image_results:
                return image_results
            # Fallback to scraping if APIs fail
            image_results = await self._search_images(topic, num_images)
            if image_results:
                return image_results
            # Final fallback to Unsplash API if scraping fails
            return await self._get_fallback_images(topic, num_images)
        except aiohttp.ClientError as e:
            logger.error(f"Network error while fetching images: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in image scraping: {e}")
            return None

    async def _get_images_from_apis(self, topic: str, num_images: int) -> Optional[Dict[str, Union[str, List[str]]]]:
        # Try Unsplash API
        if APIConfig.UNSPLASH_ACCESS_KEY:
            try:
                async with self.session.get(
                    f"https://api.unsplash.com/search/photos?query={quote_plus(topic)}&per_page={num_images}",
                    headers={"Authorization": f"Client-ID {APIConfig.UNSPLASH_ACCESS_KEY}"}) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and isinstance(data, dict) and 'results' in data and data['results'] is not None:
                            images = []
                            for photo in data['results']:
                                if photo and isinstance(photo, dict):
                                    urls = None
                                    if isinstance(photo, dict):
                                        urls = photo.get('urls')
                                    if urls is not None and isinstance(urls, dict):
                                        regular_url = urls.get('regular')
                                        if regular_url is not None:
                                            images.append(regular_url)
                                    else:
                                        logger.debug(f"Photo urls missing or invalid: {photo}")



                            if images:
                                return {'topic': topic, 'images': images[:num_images]}
                        else:
                            logger.error("Unsplash API returned no results or invalid data")
                    else:
                        logger.error(f"Unsplash API returned status {response.status}")
            except Exception as e:
                logger.error(f"Unsplash API error: {e}")

        # Try Pixabay API
        if APIConfig.PIXABAY_API_KEY:
            try:
                async with self.session.get(
                    f"https://pixabay.com/api/?key={APIConfig.PIXABAY_API_KEY}&q={quote_plus(topic)}&image_type=photo&per_page={num_images}"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and isinstance(data, dict) and 'hits' in data and data['hits'] is not None:
                            images = []
                            for hit in data['hits']:
                                if hit and isinstance(hit, dict):
                                    url = None
                                    if isinstance(hit, dict):
                                        url = hit.get('webformatURL')
                                    if url is not None:
                                        images.append(url)
                                    else:
                                        logger.debug(f"Hit webformatURL missing or invalid: {hit}")



                            if images:
                                return {'topic': topic, 'images': images[:num_images]}
                        else:
                            logger.error("Pixabay API returned no hits or invalid data")
                    else:
                        logger.error(f"Pixabay API returned status {response.status}")
            except Exception as e:
                logger.error(f"Pixabay API error: {e}")

        # Try Pexels API
        if APIConfig.PEXELS_API_KEY:
            try:
                async with self.session.get(
                    f"https://api.pexels.com/v1/search?query={quote_plus(topic)}&per_page={num_images}",
                    headers={"Authorization": APIConfig.PEXELS_API_KEY}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and isinstance(data, dict) and 'photos' in data and data['photos'] is not None:
                            images = []
                            for photo in data['photos']:
                                if photo and isinstance(photo, dict):
                                    src = None
                                    if isinstance(photo, dict):
                                        src = photo.get('src')
                                    if src is not None and isinstance(src, dict):
                                        medium_url = src.get('medium')
                                        if medium_url is not None:
                                            images.append(medium_url)
                                    else:
                                        logger.debug(f"Photo src missing or invalid: {photo}")



                            if images:
                                return {'topic': topic, 'images': images[:num_images]}
                        else:
                            logger.error("Pexels API returned no photos or invalid data")
                    else:
                        logger.error(f"Pexels API returned status {response.status}")
            except Exception as e:
                logger.error(f"Pexels API error: {e}")

        return None

    async def _search_images(self, topic: str, num_images: int) -> Optional[Dict[str, Union[str, List[str]]]]:
        search_query = self._prepare_search_query(topic)
        try:
            async with self.session.get(f"https://www.google.com/search?q={search_query}&tbm=isch",
                                        headers=self.headers) as response:
                if response.status != 200:
                    return None
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                images = []
                for img_tag in soup.find_all(['img', 'div']):
                    if not isinstance(img_tag, Tag):
                        continue
                    img_url = None
                    for attr in ['src', 'data-src', 'data-original']:
                        if img_tag is None:
                            continue
                        get_method = getattr(img_tag, 'get', None)
                        if get_method is None:
                            continue
                        if callable(get_method):
                            img_url = get_method(attr)
                            if img_url is not None:
                                break
                    if img_url is not None:
                        images.append(img_url)
                    if len(images) >= num_images:
                        break




                return {'topic': topic, 'images': images[:num_images]}
        except Exception as e:
            logger.error(f"Error during image scraping: {e}")
            return None
