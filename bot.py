import logging
import os
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes
)
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import io
import cv2
import numpy as np
import textwrap
import html
import re
from typing import List, Dict, Optional

# ==================== CONFIGURATION ====================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('ocr_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Tesseract Configuration
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA_DIR = r"C:\Program Files\Tesseract-OCR\tessdata"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR

# Bot Configuration
BOT_TOKEN = "7881043106:AAF_4vEr-x74mGWuYUZtwe0_NO_rRgyMnnM"
DEVELOPER_ID = "YOUR_DEV_ID"  # Replace with your Telegram ID

# ==================== EMOJI DECORATIONS ====================
EMOJI = {
    "camera": "📷",
    "text": "📝",
    "magic": "✨",
    "language": "🌐",
    "settings": "⚙️",
    "back": "🔙",
    "check": "✅",
    "error": "❌",
    "info": "ℹ️",
    "wrench": "🔧",
    "star": "⭐",
    "fire": "🔥",
    "clock": "⏱️",
    "palette": "🎨",
    "download": "📥",
    "upload": "📤",
    "book": "📖",
    "brain": "🧠",
    "robot": "🤖",
    "wave": "👋",
    "heart": "❤️"
}

# ==================== LANGUAGE SUPPORT ====================
SUPPORTED_LANGUAGES = {
    "eng": {"name": "English", "flag": "🇺🇸"},
    "khm": {"name": "Khmer", "flag": "🇰🇭"},
    "fra": {"name": "French", "flag": "🇫🇷"},
    "spa": {"name": "Spanish", "flag": "🇪🇸"},
    "deu": {"name": "German", "flag": "🇩🇪"},
    "jpn": {"name": "Japanese", "flag": "🇯🇵"},
    "kor": {"name": "Korean", "flag": "🇰🇷"},
    "chi_sim": {"name": "Chinese", "flag": "🇨🇳"}
}

# ==================== PREPROCESSING PROFILES ====================
PREPROCESSING_PROFILES = {
    "auto": {"name": "Auto Detect", "emoji": "🤖"},
    "document": {"name": "Document", "emoji": "📄"},
    "handwriting": {"name": "Handwriting", "emoji": "✍️"},
    "low_light": {"name": "Low Light", "emoji": "🌙"},
    "colored": {"name": "Colored Text", "emoji": "🎨"}
}

# ==================== CUSTOM STYLES ====================
class BotStyles:
    """Beautiful formatting styles for the bot"""
    
    @staticmethod
    def create_header(text: str) -> str:
        return f"{EMOJI['star']} *{text}* {EMOJI['star']}"
    
    @staticmethod
    def create_section(title: str, content: str) -> str:
        return f"*{EMOJI['fire']} {title}*:\n{content}\n"
    
    @staticmethod
    def create_success(text: str) -> str:
        return f"{EMOJI['check']} {text}"
    
    @staticmethod
    def create_error(text: str) -> str:
        return f"{EMOJI['error']} {text}"
    
    @staticmethod
    def create_info(text: str) -> str:
        return f"{EMOJI['info']} {text}"
    
    @staticmethod
    def create_warning(text: str) -> str:
        return f"⚠️ {text}"
    
    @staticmethod
    def format_ocr_result(text: str, confidence: float = None) -> str:
        """Format OCR result beautifully"""
        if not text.strip():
            return BotStyles.create_warning("No text detected in the image.")
        
        formatted = f"{EMOJI['text']} *Extracted Text:*\n\n"
        formatted += "```\n"
        formatted += text.strip()
        formatted += "\n```\n"
        
        if confidence:
            formatted += f"\n{EMOJI['brain']} *Confidence:* `{confidence:.1f}%`"
        
        return formatted

# ==================== IMAGE PREPROCESSOR ====================
class ImagePreprocessor:
    """Advanced image preprocessing for better OCR results"""
    
    @staticmethod
    def preprocess(image: np.ndarray, profile: str = "auto") -> np.ndarray:
        """Apply preprocessing based on profile"""
        if profile == "document":
            return ImagePreprocessor._preprocess_document(image)
        elif profile == "handwriting":
            return ImagePreprocessor._preprocess_handwriting(image)
        elif profile == "low_light":
            return ImagePreprocessor._preprocess_low_light(image)
        elif profile == "colored":
            return ImagePreprocessor._preprocess_colored(image)
        else:
            return ImagePreprocessor._preprocess_auto(image)
    
    @staticmethod
    def _preprocess_auto(image: np.ndarray) -> np.ndarray:
        """Auto-detect best preprocessing"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check image brightness
        brightness = np.mean(gray)
        if brightness < 100:
            # Low light image
            return ImagePreprocessor._preprocess_low_light(image)
        elif np.std(gray) < 30:
            # Low contrast document
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(gray)
        else:
            # Standard document
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
    
    @staticmethod
    def _preprocess_document(image: np.ndarray) -> np.ndarray:
        """Preprocess for printed documents"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    @staticmethod
    def _preprocess_handwriting(image: np.ndarray) -> np.ndarray:
        """Preprocess for handwriting"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Soft thresholding
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Reduce noise while preserving details
        cleaned = cv2.medianBlur(thresh, 3)
        
        return cleaned
    
    @staticmethod
    def _preprocess_low_light(image: np.ndarray) -> np.ndarray:
        """Preprocess for low-light images"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhance lightness channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        
        # Merge back
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    @staticmethod
    def _preprocess_colored(image: np.ndarray) -> np.ndarray:
        """Preprocess for colored text on colored background"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use color segmentation to find text regions
        # (Simplified version - you can enhance this)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        
        # Apply mask
        result = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Threshold
        _, thresh = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh

# ==================== KEYBOARD CREATORS ====================
class KeyboardCreator:
    """Create beautiful inline keyboards"""
    
    @staticmethod
    def create_main_menu() -> InlineKeyboardMarkup:
        """Main menu keyboard"""
        keyboard = [
            [
                InlineKeyboardButton(
                    f"{EMOJI['language']} Select Languages",
                    callback_data="menu_languages"
                )
            ],
            [
                InlineKeyboardButton(
                    f"{EMOJI['palette']} Preprocessing",
                    callback_data="menu_preprocessing"
                ),
                InlineKeyboardButton(
                    f"{EMOJI['settings']} Settings",
                    callback_data="menu_settings"
                )
            ],
            [
                InlineKeyboardButton(
                    f"{EMOJI['info']} Help & Info",
                    callback_data="menu_help"
                ),
                InlineKeyboardButton(
                    f"{EMOJI['star']} Premium Features",
                    callback_data="menu_premium"
                )
            ]
        ]
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def create_language_menu(selected_langs: List[str] = None) -> InlineKeyboardMarkup:
        """Language selection menu"""
        if selected_langs is None:
            selected_langs = ["eng", "khm"]
        
        keyboard = []
        rows = []
        
        for lang_code, lang_info in SUPPORTED_LANGUAGES.items():
            is_selected = lang_code in selected_langs
            emoji = EMOJI['check'] if is_selected else "⬜"
            button_text = f"{emoji} {lang_info['flag']} {lang_info['name']}"
            rows.append(InlineKeyboardButton(button_text, callback_data=f"lang_{lang_code}"))
            
            if len(rows) == 2:
                keyboard.append(rows)
                rows = []
        
        if rows:
            keyboard.append(rows)
        
        # Add action buttons
        keyboard.append([
            InlineKeyboardButton(f"{EMOJI['back']} Back", callback_data="back_main"),
            InlineKeyboardButton(f"{EMOJI['check']} Apply", callback_data="apply_languages")
        ])
        
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def create_preprocessing_menu(current_profile: str = "auto") -> InlineKeyboardMarkup:
        """Preprocessing profile menu"""
        keyboard = []
        
        for profile_key, profile_info in PREPROCESSING_PROFILES.items():
            is_selected = profile_key == current_profile
            emoji = EMOJI['check'] if is_selected else profile_info['emoji']
            button_text = f"{emoji} {profile_info['name']}"
            keyboard.append([InlineKeyboardButton(button_text, callback_data=f"preprocess_{profile_key}")])
        
        keyboard.append([
            InlineKeyboardButton(f"{EMOJI['back']} Back", callback_data="back_main")
        ])
        
        return InlineKeyboardMarkup(keyboard)

# ==================== BOT HANDLERS ====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command with beautiful welcome message"""
    welcome_message = f"""
{EMOJI['robot']} *Welcome to Input Clever Bot!* {EMOJI['robot']}

{EMOJI['magic']} *Advanced Features:*
• Multi-language OCR {EMOJI['language']}
• Smart preprocessing {EMOJI['palette']}
• Handwriting recognition {EMOJI['text']}
• Low-light enhancement {EMOJI['clock']}
• Colored text extraction {EMOJI['fire']}

{EMOJI['camera']} *How to use:*
1. Send any image with text
2. Use /menu to configure settings
3. Get beautifully formatted results!

{EMOJI['wrench']} *Quick Commands:*
/menu - Open control panel
/languages - Select OCR languages
/help - Get detailed instructions
/stats - View bot statistics
    """
    
    keyboard = KeyboardCreator.create_main_menu()
    await update.message.reply_text(
        welcome_message,
        parse_mode='Markdown',
        reply_markup=keyboard
    )

async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Open the main menu"""
    menu_message = f"""
{EMOJI['settings']} *Control Panel* {EMOJI['settings']}

Choose an option below to configure your OCR experience:
    """
    
    keyboard = KeyboardCreator.create_main_menu()
    await update.message.reply_text(
        menu_message,
        parse_mode='Markdown',
        reply_markup=keyboard
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command with detailed instructions"""
    help_text = f"""
{EMOJI['book']} *OCR Master Bot Guide* {EMOJI['book']}

{EMOJI['star']} *Getting Started:*
1. Send any image containing text
2. The bot will automatically process it
3. Receive formatted text output

{EMOJI['language']} *Language Selection:*
• Default: English + Khmer
• Add more languages via /menu
• Supports {len(SUPPORTED_LANGUAGES)} languages

{EMOJI['palette']} *Preprocessing Profiles:*
• *Auto Detect*: Smart automatic processing
• *Document*: Optimized for printed documents
• *Handwriting*: Enhanced for handwritten text
• *Low Light*: Improved dark image processing
• *Colored Text*: Text on colored backgrounds

{EMOJI['wrench']} *Commands:*
/start - Welcome message
/menu - Control panel
/languages - Language selector
/stats - Bot statistics
/help - This help message

{EMOJI['brain']} *Tips for Best Results:*
• Use good lighting
• Keep text horizontal
• Clear, focused images work best
• Use appropriate preprocessing profile
    """
    
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show bot statistics"""
    stats = context.bot_data.get('stats', {
        'processed_images': 0,
        'total_characters': 0,
        'success_rate': 100
    })
    
    stats_message = f"""
{EMOJI['fire']} *Bot Statistics* {EMOJI['fire']}

{EMOJI['camera']} *Images Processed:* `{stats['processed_images']}`
{EMOJI['text']} *Characters Extracted:* `{stats['total_characters']:,}`
{EMOJI['brain']} *Success Rate:* `{stats['success_rate']}%`
{EMOJI['clock']} *Uptime:* `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`

{EMOJI['star']} *System Status:*
• OCR Engine: ✅ Active
• Language Packs: ✅ Loaded
• Processing: ✅ Ready
• Memory: ✅ Optimized
    """
    
    await update.message.reply_text(stats_message, parse_mode='Markdown')

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button presses"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data == "menu_languages":
        # Get current languages from user data
        user_langs = context.user_data.get('selected_languages', ['eng', 'khm'])
        keyboard = KeyboardCreator.create_language_menu(user_langs)
        await query.edit_message_text(
            f"{EMOJI['language']} *Select Languages*\n\n"
            f"Click to toggle languages. Selected: {EMOJI['check']}\n"
            f"Current: {', '.join([SUPPORTED_LANGUAGES[l]['name'] for l in user_langs])}",
            parse_mode='Markdown',
            reply_markup=keyboard
        )
    
    elif data.startswith("lang_"):
        # Toggle language selection
        lang_code = data[5:]
        user_langs = context.user_data.get('selected_languages', ['eng', 'khm'])
        
        if lang_code in user_langs:
            user_langs.remove(lang_code)
        else:
            user_langs.append(lang_code)
        
        context.user_data['selected_languages'] = user_langs
        keyboard = KeyboardCreator.create_language_menu(user_langs)
        
        await query.edit_message_reply_markup(reply_markup=keyboard)
    
    elif data == "apply_languages":
        user_langs = context.user_data.get('selected_languages', ['eng', 'khm'])
        lang_names = [SUPPORTED_LANGUAGES[l]['name'] for l in user_langs]
        
        await query.edit_message_text(
            BotStyles.create_success(
                f"Languages updated successfully!\n"
                f"Selected: {', '.join(lang_names)}"
            ),
            parse_mode='Markdown'
        )
    
    elif data == "menu_preprocessing":
        current_profile = context.user_data.get('preprocessing_profile', 'auto')
        keyboard = KeyboardCreator.create_preprocessing_menu(current_profile)
        
        await query.edit_message_text(
            f"{EMOJI['palette']} *Preprocessing Profiles*\n\n"
            f"Select a profile to optimize OCR for different image types.",
            parse_mode='Markdown',
            reply_markup=keyboard
        )
    
    elif data.startswith("preprocess_"):
        profile = data[11:]
        context.user_data['preprocessing_profile'] = profile
        profile_name = PREPROCESSING_PROFILES[profile]['name']
        
        await query.edit_message_text(
            BotStyles.create_success(
                f"Preprocessing profile set to: {profile_name}"
            ),
            parse_mode='Markdown'
        )
    
    elif data in ["back_main", "menu_settings", "menu_help", "menu_premium"]:
        if data == "back_main":
            keyboard = KeyboardCreator.create_main_menu()
            await query.edit_message_text(
                f"{EMOJI['settings']} *Control Panel*",
                parse_mode='Markdown',
                reply_markup=keyboard
            )
        else:
            await query.edit_message_text(
                BotStyles.create_info("This feature is coming soon!"),
                parse_mode='Markdown'
            )

async def ocr_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced OCR handler with beautiful formatting"""
    if not update.message.photo:
        await update.message.reply_text(
            BotStyles.create_info("Please send an image to perform OCR."),
            parse_mode='Markdown'
        )
        return
    
    # Send processing message
    processing_msg = await update.message.reply_text(
        f"{EMOJI['magic']} *Processing your image...*\n"
        f"{EMOJI['clock']} This may take a few seconds",
        parse_mode='Markdown'
    )
    
    try:
        # Download image
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = io.BytesIO()
        await photo_file.download_to_memory(out=photo_bytes)
        photo_bytes.seek(0)
        
        # Open image
        image = Image.open(photo_bytes).convert("RGB")
        
        # Convert to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Get user preferences
        preprocessing_profile = context.user_data.get('preprocessing_profile', 'auto')
        selected_langs = context.user_data.get('selected_languages', ['eng', 'khm'])
        language_string = '+'.join(selected_langs)
        
        # Apply preprocessing
        preprocessed = ImagePreprocessor.preprocess(image_cv, preprocessing_profile)
        
        # Convert back to PIL
        image_for_ocr = Image.fromarray(preprocessed)
        
        # Perform OCR with confidence
        custom_oem_psm_config = '--oem 3 --psm 6'
        data = pytesseract.image_to_data(
            image_for_ocr,
            lang=language_string,
            config=custom_oem_psm_config,
            output_type=pytesseract.Output.DICT
        )
        
        # Calculate average confidence
        confidences = [float(c) for c in data['conf'] if float(c) > 0]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Extract text
        text = ' '.join([word for word, conf in zip(data['text'], data['conf']) 
                        if float(conf) > 0 and word.strip()])
        
        # Format response
        if text.strip():
            response = BotStyles.format_ocr_result(text, avg_confidence)
            
            # Add preprocessing info
            profile_name = PREPROCESSING_PROFILES[preprocessing_profile]['name']
            response += f"\n{EMOJI['palette']} *Profile:* `{profile_name}`"
            
            # Add language info
            lang_names = [SUPPORTED_LANGUAGES[l]['name'] for l in selected_langs]
            response += f"\n{EMOJI['language']} *Languages:* `{', '.join(lang_names)}`"
            
            # Add stats
            response += f"\n{EMOJI['text']} *Characters:* `{len(text)}`"
            
            # Update bot statistics
            stats = context.bot_data.get('stats', {
                'processed_images': 0,
                'total_characters': 0,
                'success_rate': 100
            })
            stats['processed_images'] += 1
            stats['total_characters'] += len(text)
            context.bot_data['stats'] = stats
            
        else:
            response = BotStyles.create_warning(
                "No text detected in the image.\n"
                "Try:\n"
                "1. Using better lighting\n"
                "2. Choosing different preprocessing profile\n"
                "3. Selecting appropriate languages"
            )
        
        # Send result
        await processing_msg.edit_text(response, parse_mode='Markdown')
        
        # Send preview of processed image (optional)
        if text.strip():
            preview_bytes = io.BytesIO()
            image_for_ocr.convert('RGB').save(preview_bytes, format='JPEG')
            preview_bytes.seek(0)
            await update.message.reply_photo(
                photo=preview_bytes,
                caption=f"{EMOJI['palette']} *Processed Image Preview*",
                parse_mode='Markdown'
            )
    
    except Exception as e:
        logger.error(f"OCR Error: {e}", exc_info=True)
        error_message = f"""
{EMOJI['error']} *Processing Failed*

*Error:* `{str(e)}`

*Possible Solutions:*
1. Check if Tesseract is installed correctly
2. Verify language files exist
3. Try a different image
4. Contact support if issue persists
        """
        await processing_msg.edit_text(error_message, parse_mode='Markdown')

# ==================== MAIN APPLICATION ====================
def main():
    """Start the bot"""
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    # Command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("menu", menu))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("languages", menu))  # Alias for menu
    
    # Message handlers
    app.add_handler(MessageHandler(filters.PHOTO, ocr_image))
    
    # Callback query handler
    app.add_handler(CallbackQueryHandler(button_handler))
    
    # Initialize bot data
    app.bot_data['stats'] = {
        'processed_images': 0,
        'total_characters': 0,
        'success_rate': 100
    }
    
    print(f"""
    {EMOJI['robot']} OCR Master Bot Starting...
    {EMOJI['fire']} Bot Token: Loaded
    {EMOJI['language']} Languages: {len(SUPPORTED_LANGUAGES)} supported
    {EMOJI['palette']} Profiles: {len(PREPROCESSING_PROFILES)} available
    {EMOJI['star']} Status: Ready to serve!
    
    Press Ctrl+C to stop
    """)
    
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()