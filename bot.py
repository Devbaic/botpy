import logging
import os
import io
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from PIL import Image
import pytesseract
import cv2
import numpy as np

# ==================== CONFIG ====================

BOT_TOKEN = os.getenv("7881043106:AAF_4vEr-x74mGWuYUZtwe0_NO_rRgyMnnM")  # Set in Render Environment Variables

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable not set")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Tesseract (Linux / Render)
pytesseract.pytesseract.tesseract_cmd = "tesseract"

# ==================== BOT COMMANDS ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *OCR Bot is running!*\n\n"
        "📷 Send me an image and I will extract text.\n"
        "/help – instructions",
        parse_mode="Markdown",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *How to use*\n\n"
        "1️⃣ Send a photo with text\n"
        "2️⃣ Wait a moment\n"
        "3️⃣ Get extracted text\n\n"
        "Supported: English OCR",
        parse_mode="Markdown",
    )


# ==================== OCR HANDLER ====================

async def ocr_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("✨ Processing image...")

    try:
        photo = update.message.photo[-1]
        file = await photo.get_file()

        bio = io.BytesIO()
        await file.download_to_memory(bio)
        bio.seek(0)

        image = Image.open(bio).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Simple threshold for better OCR
        _, thresh = cv2.threshold(
            image_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        text = pytesseract.image_to_string(thresh, lang="eng")

        if not text.strip():
            await msg.edit_text("❌ No text detected.")
            return

        await msg.edit_text(
            f"📝 *Extracted Text:*\n\n```\n{text.strip()}\n```",
            parse_mode="Markdown",
        )

    except Exception as e:
        logger.exception("OCR failed")
        await msg.edit_text(f"❌ Error:\n`{e}`", parse_mode="Markdown")


# ==================== MAIN ====================

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.PHOTO, ocr_image))

    logger.info("🤖 Bot started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
