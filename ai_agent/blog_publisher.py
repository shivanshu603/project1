import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from utils import logger
from models import Article
import aiohttp
from config import Config
import re
import json
from utils.content_humanizer import ContentHumanizer
import time  # Added missing import for time

from category_detector import CategoryDetector
from content_formatter import ContentFormatter

class BlogPublisher:
    def __init__(self, wp_url: str, wp_username: str, wp_password: str):
        self.wp_url = wp_url or Config.WORDPRESS_SITE_URL
        self.wp_username = wp_username or Config.WORDPRESS_USERNAME
        self.wp_password = wp_password or Config.WORDPRESS_PASSWORD
        self.session = None
        self.retry_count = 0
        self.max_retries = Config.MAX_RETRIES
        self.retry_delay = Config.RETRY_DELAY
        
        # Initialize the content humanizer
        try:
            self.humanizer = ContentHumanizer()
            logger.info("ContentHumanizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ContentHumanizer: {e}")
            self.humanizer = None

        # Initialize CategoryDetector and ContentFormatter
        self.category_detector = CategoryDetector()
        self.content_formatter = ContentFormatter()



    async def cleanup(self):
        """Cleanup resources such as aiohttp session"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
                logger.info("Closed aiohttp session successfully")
        except Exception as e:
            logger.error(f"Error during BlogPublisher cleanup: {e}")


    async def _init_session(self) -> bool:
        """Initialize aiohttp session with retry logic"""
        for attempt in range(self.max_retries + 1):
            try:
                if self.session is None:
                    self.session = aiohttp.ClientSession()
                
                async with self.session.get(f"{self.wp_url}/wp-json/wp/v2/posts", auth=aiohttp.BasicAuth(self.wp_username, self.wp_password)) as response:
                    if response.status == 200:
                        logger.info("Successfully initialized WordPress session")
                        return True
                    else:
                        logger.error(f"Failed to initialize session, status: {response.status}")
                
            except Exception as e:
                logger.error(f"Session initialization error (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if self.session:
                    await self.session.close()
                    self.session = None
                
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        return False

    async def verify_connection(self) -> bool:
        """Verify WordPress connection with enhanced error handling"""
        if not await self._init_session() or not self.session:
            return False

        try:
            test_url = f"{self.wp_url}/wp-json/"
            async with self.session.get(test_url) as response:
                if response.status == 200:
                    logger.info("WordPress connection verified successfully")
                    return True
                else:
                    error_details = await self._get_error_details(response)
                    logger.error(f"WordPress connection failed: {error_details}")
                    return False
        except Exception as e:
            logger.error(f"Error verifying WordPress connection: {str(e)}")
            return False

    async def upload_image_from_url(self, image_url: str) -> Optional[int]:
        """Download image from URL and upload to WordPress media library, return media ID"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to download image from {image_url}, status {response.status}")
                        return None
                    image_data = await response.read()
                    content_type = response.headers.get('Content-Type', 'image/jpeg')

                    # Validate filename and extension
                    filename = image_url.split("/")[-1].split("?")[0]
                    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
                    if not filename or not any(filename.lower().endswith(ext) for ext in valid_extensions):
                        filename = f"image_{int(time.time())}.jpg"
                        content_type = 'image/jpeg'  # Force content type to jpeg if unknown or invalid

                    headers = {
                        'Content-Disposition': f'attachment; filename="{filename}"',
                        'Content-Type': content_type
                    }

                    logger.info(f"Uploading image with filename: {filename} and content type: {content_type}")

                    auth = aiohttp.BasicAuth(self.wp_username, self.wp_password)
                    media_url = f"{self.wp_url}/wp-json/wp/v2/media"

                    async with aiohttp.ClientSession() as upload_session:
                        async with upload_session.post(
                            media_url,
                            data=image_data,
                            headers=headers,
                            auth=auth
                        ) as upload_response:
                            if upload_response.status in (200, 201):
                                media = await upload_response.json()
                                media_id = media.get('id')
                                logger.info(f"Uploaded image {filename} with media ID {media_id}")
                                return media_id
                            else:
                                error_text = await upload_response.text()
                                logger.error(f"Failed to upload image {filename}, status {upload_response.status}, error: {error_text}")
                                return None
        except Exception as e:
            logger.error(f"Exception in upload_image_from_url: {e}")
            return None


    async def publish_article(self, article: Article) -> bool:
        """Publish article to WordPress with categories and tags"""
        try:
            if not article:
                logger.error("No article provided")
                return False

            logger.info(f"Starting to publish article: {article.title}")

            # Handle categories using CategoryDetector if missing
            if not article.categories:
                detected_categories = self.category_detector.detect_categories(
                    article.title,
                    article.content,
                    getattr(article, 'keywords', None)
                )
                article.categories = detected_categories
                logger.info(f"Detected categories: {article.categories}")

            # Handle tags using CategoryDetector if missing
            if not article.tags:
                detected_tags = self.category_detector.extract_tags_from_title(article.title)
                article.tags = detected_tags
                logger.info(f"Detected tags: {article.tags}")

            # Handle images first
            image_ids = []
            if hasattr(article, 'images') and article.images:
                for image in article.images:
                    if isinstance(image, dict) and 'url' in image:
                        try:
                            logger.info(f"Downloading and uploading image from URL: {image['url']}")
                            media_id = await self.upload_image_from_url(image['url'])
                            if media_id:
                                image_ids.append(media_id)
                                image['id'] = media_id

                                # Update alt text
                                await self._update_media_alt_text(media_id, article.title)
                                logger.info(f"Updated alt text for media ID: {media_id}")
                        except Exception as e:
                            logger.error(f"Failed to upload image {image['url']}: {e}")

            # Format content using ContentFormatter
            formatted_content = self.content_formatter.format_article(article.content)

            # Convert markdown to WordPress format
            formatted_content = self._convert_markdown_to_gutenberg(formatted_content)
            
            # If we have images, append them to content
            if image_ids:
                for media_id in image_ids:
                    # Get image URL from WordPress
                    image_url = await self._get_media_url(media_id)
                    if image_url:
                        logger.info(f"Retrieved WordPress URL for media {media_id}: {image_url}")
                        formatted_content += f'\n<!-- wp:image {{"id":{media_id}}} -->\n'
                        formatted_content += f'<figure class="wp-block-image"><img src="{image_url}" alt="{article.title}"/></figure>\n'
                        formatted_content += '<!-- /wp:image -->\n'
                        logger.info(f"Appended image at end of content: {media_id}")


            # Use first image as featured image
            if image_ids:
                article.featured_image_id = image_ids[0]
                logger.info("Using first image as featured image")

            # Prepare post data
            # Convert tag names to tag IDs
            tag_ids = []
            if article.tags:
                for tag in article.tags:
                    tag_id = await self._get_or_create_tag(tag)
                    if tag_id:
                        tag_ids.append(tag_id)

            post_data = {
                'title': article.title,
                'content': formatted_content,
                'status': 'publish',
                'categories': article.categories or [1],  # Default to Uncategorized if no categories
                'tags': tag_ids,
                'featured_media': article.featured_image_id if hasattr(article, 'featured_image_id') else 0
            }


            # Add excerpt/meta description if present
            if hasattr(article, 'meta_description') and article.meta_description:
                post_data['excerpt'] = {'raw': article.meta_description}

            # Create WordPress post
            logger.info("Creating WordPress post")
            auth = aiohttp.BasicAuth(self.wp_username, self.wp_password)
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.wp_url}/wp-json/wp/v2/posts", json=post_data, auth=auth) as response:
                    if response.status in (200, 201):
                        post = await response.json()
                        post_id = post.get('id')
                        if post_id:
                            logger.info(f"Post created with ID: {post_id}")

                            # Add categories and tags
                            logger.info(f"Adding categories and tags to post {post_id}")
                            await self._add_categories_and_tags(post_id, article.categories, article.tags)

                            return True
                        else:
                            logger.error("Post creation response missing ID")
                            return False
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to create post. Status: {response.status}, Error: {error_text}")
                        return False

        except Exception as e:
            logger.error(f"Error publishing article: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    async def _add_categories_and_tags(self, post_id: int, categories: List[int], tags: List[str]) -> bool:
        """Add categories and tags to a WordPress post"""
        if not self.session:
            await self._init_session()
        if not self.session:
            logger.error("Cannot add categories and tags: session not initialized")
            return False
        post_endpoint = f"{self.wp_url}/wp-json/wp/v2/posts/{post_id}"
        auth = aiohttp.BasicAuth(self.wp_username, self.wp_password)
        payload = {}
        if categories:
            payload['categories'] = categories
        if tags:
            # Convert tag names to tag IDs asynchronously
            tag_ids = []
            for tag in tags:
                tag_id = await self._get_or_create_tag(tag)
                if tag_id:
                    tag_ids.append(tag_id)
            payload['tags'] = tag_ids
        if not payload:
            logger.info("No categories or tags to add")
            return True
        try:
            async with self.session.post(post_endpoint, json=payload, auth=auth) as response:
                if response.status in (200, 201):
                    logger.info(f"Successfully added categories and tags to post {post_id}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to add categories and tags to post {post_id}, status {response.status}, error: {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Exception adding categories and tags to post {post_id}: {e}")
            return False


    async def _get_or_create_tag(self, tag_name: str) -> Optional[int]:
        """Get existing tag ID by name or create a new tag in WordPress and return its ID"""
        if not self.session:
            await self._init_session()
        if not self.session:
            logger.error("Cannot get or create tag: session not initialized")
            return None
        tags_endpoint = f"{self.wp_url}/wp-json/wp/v2/tags"
        auth = aiohttp.BasicAuth(self.wp_username, self.wp_password)
        try:
            # Search for existing tag by name
            params = {'search': tag_name}
            async with self.session.get(tags_endpoint, params=params, auth=auth) as response:
                if response.status == 200:
                    tags = await response.json()
                    for tag in tags:
                        if tag.get('name').lower() == tag_name.lower():
                            return tag.get('id')
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to search tags for '{tag_name}', status {response.status}, error: {error_text}")
                    return None

            # Tag not found, create new tag
            payload = {'name': tag_name}
            async with self.session.post(tags_endpoint, json=payload, auth=auth) as response:
                if response.status in (200, 201):
                    tag = await response.json()
                    tag_id = tag.get('id')
                    logger.info(f"Created new tag '{tag_name}' with ID {tag_id}")
                    return tag_id
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to create tag '{tag_name}', status {response.status}, error: {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Exception in _get_or_create_tag for '{tag_name}': {e}")
            return None

    def _convert_markdown_to_gutenberg(self, content: str) -> str:
        """Convert markdown headings and formatting to WordPress Gutenberg blocks"""
        try:
            # Split content into lines for processing
            lines = content.split('\n')
            gutenberg_content = []
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Check if this is a markdown heading
                heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
                if heading_match:
                    # This is a heading - convert to WordPress heading block
                    hashes = heading_match.group(1)
                    heading_text = heading_match.group(2).strip()
                    heading_level = len(hashes)
                    
                    # Create WordPress heading block
                    gutenberg_content.append(f'<!-- wp:heading {{"level":{heading_level}}} -->')
                    gutenberg_content.append(f'<h{heading_level}>{heading_text}</h{heading_level}>')
                    gutenberg_content.append(f'<!-- /wp:heading -->')
                    gutenberg_content.append('')  # Add empty line for spacing
                
                # Check if this is a list item
                elif line.startswith('- ') or line.startswith('* '):
                    # This is a list item - collect all list items
                    list_items = []
                    list_type = 'ul'  # unordered list
                    while i < len(lines) and (lines[i].strip().startswith('- ') or lines[i].strip().startswith('* ')):
                        item_text = lines[i].strip()[2:].strip()
                        list_items.append(f'<li>{item_text}</li>')
                        i += 1
                    
                    # Create WordPress list block
                    gutenberg_content.append(f'<!-- wp:list -->')
                    gutenberg_content.append(f'<{list_type}>')
                    gutenberg_content.extend(list_items)
                    gutenberg_content.append(f'</{list_type}>')
                    gutenberg_content.append(f'<!-- /wp:list -->')
                    gutenberg_content.append('')  # Add empty line for spacing
                    
                    # Continue without incrementing i since we already advanced it in the loop
                    continue
                
                # Check if this is a paragraph
                elif line and not line.startswith('<!--'):
                    # This is a paragraph - convert to WordPress paragraph block
                    gutenberg_content.append(f'<!-- wp:paragraph -->')
                    gutenberg_content.append(f'<p>{line}</p>')
                    gutenberg_content.append(f'<!-- /wp:paragraph -->')
                    gutenberg_content.append('')  # Add empty line for spacing
                
                # If it's already a WordPress block or empty line, keep as is
                else:
                    gutenberg_content.append(line)
                
                i += 1
            
            # Join all lines back together
            return '\n'.join(gutenberg_content)
            
        except Exception as e:
            logger.error(f"Error converting markdown to Gutenberg: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return original content if conversion fails
            return content

    async def _update_media_alt_text(self, media_id: int, alt_text: str) -> bool:
        """Update the alt text of a media item in WordPress"""
        if not self.session:
            await self._init_session()
        if not self.session:
            logger.error("Cannot update media alt text: session not initialized")
            return False
        media_endpoint = f"{self.wp_url}/wp-json/wp/v2/media/{media_id}"
        payload = {
            "alt_text": alt_text
        }
        try:
            auth = aiohttp.BasicAuth(self.wp_username, self.wp_password)
            async with self.session.post(media_endpoint, json=payload, auth=auth) as response:
                if response.status in (200, 201):
                    logger.info(f"Successfully updated alt text for media ID {media_id}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to update alt text for media ID {media_id}, status {response.status}, error: {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Exception updating alt text for media ID {media_id}: {e}")
            return False

    async def _get_media_url(self, media_id: int) -> Optional[str]:
        """Retrieve the source URL of a media item from WordPress"""
        if not self.session:
            await self._init_session()
        if not self.session:
            logger.error("Cannot get media URL: session not initialized")
            return None
        media_endpoint = f"{self.wp_url}/wp-json/wp/v2/media/{media_id}"
        try:
            auth = aiohttp.BasicAuth(self.wp_username, self.wp_password)
            async with self.session.get(media_endpoint, auth=auth) as response:
                if response.status == 200:
                    media = await response.json()
                    source_url = media.get('source_url')
                    if source_url:
                        return source_url
                    else:
                        logger.error(f"Media ID {media_id} has no source_url field")
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get media URL for media ID {media_id}, status {response.status}, error: {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Exception getting media URL for media ID {media_id}: {e}")
            return None
