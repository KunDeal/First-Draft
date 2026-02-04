import os
import re
print("Importing logging...")
import logging
print("Importing asyncio...")
import asyncio
print("Importing dotenv...")
from dotenv import load_dotenv
print("Importing aiogram...")
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.utils.chat_action import ChatActionSender
print("Importing chromadb...")
import chromadb
print("Importing sentence_transformers...")
from sentence_transformers import SentenceTransformer
print("Importing openai...")
from openai import AsyncOpenAI
print("Importing ingest...")
import ingest
print("Imports done.")

# Load environment variables
load_dotenv()

# Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY")
ROUTER_BASE_URL = os.getenv("ROUTER_BASE_URL")
VECTOR_DB_DIR = "vector_db"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
LOGS_DIR = "logs"

# Setup logging
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "bot.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

print("Starting bot...")

# Initialize Bot and Dispatcher
if not TELEGRAM_TOKEN:
    print("Error: TELEGRAM_TOKEN not found")
    logging.error("TELEGRAM_TOKEN not found in .env")
    exit(1)

print("Token found, initializing bot...")
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# Initialize ChromaDB and Embedding Model
try:
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    collection = chroma_client.get_or_create_collection(name="legal_rag")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logging.info("ChromaDB and Embedding Model initialized successfully.")
except Exception as e:
    logging.error(f"Initialization error: {e}")
    exit(1)

# Initialize RouterAI Client
router_client = AsyncOpenAI(
    api_key=ROUTER_API_KEY,
    base_url=ROUTER_BASE_URL
)

SYSTEM_PROMPT = """
Ты — Legal AI Analyst. Твоя задача — давать точные юридические прогнозы на основе предоставленных фрагментов (CONTEXT).

---
ВАЖНЕЙШЕЕ ПРАВИЛО ПО ССЫЛКАМ:
1. 🛑 **ЗАПРЕТ:** НИКОГДА не используй номер дела "А40-12345/13" или "А40-12345/23". Это пример!
2. 🔗 **ИСТОЧНИК:** При цитировании информации ВСЕГДА используй предоставленный 'SOURCE_ID' (например, [Дело №А40-12345/23]). НЕ ссылайся на "фрагмент 1" или "источник 1".
3. 🔗 **ССЫЛКА:** Если возможно, оформи номер дела как Markdown-ссылку: `https://kad.arbitr.ru/Card/{Case_Number}`.
   - Пример: `[Дело №А40-12854-2013](https://kad.arbitr.ru/Card/А40-12854-2013)`

---
СТРУКТУРА ОТВЕТА:
1. **Вердикт**: (Шансы высокие/низкие + краткое обоснование).
2. **Анализ**:
   - Аргумент 1 [Ссылка]
   - Аргумент 2 [Ссылка]

Твой стиль: сухой, профессиональный, юридически точный.
"""

def extract_case_number(filename):
    """
    Extracts case number from filename using regex.
    Pattern: A\d{2}-\d{3,}/?\d{2,4} (Case-insensitive, handling Cyrillic 'А' and Latin 'A').
    Example: "Дело №А40-12854-2013.docx" -> "А40-12854-2013"
    Fallback: clean filename without extension.
    """
    # Pattern to look for: A\d{2}-\d{3,}/?\d{2,4}
    # Matches A40-12345-23 or A40-12345/23
    pattern = r"([АA]\d{2}-\d{3,}[-/]\d{2,4})"
    match = re.search(pattern, filename, re.IGNORECASE)
    
    if match:
        return match.group(1).upper()
    
    # Fallback: clean filename without extension
    name = os.path.splitext(filename)[0]
    return name.replace("Delo_", "")

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Здравствуйте. Я юридический бот-аналитик. Задайте мне вопрос по арбитражной практике.")

@dp.message(Command("refresh"))
async def cmd_refresh(message: types.Message):
    """Admin command to refresh the knowledge base."""
    # In a real app, you should check for admin ID
    status_msg = await message.answer("🔄 Обновление базы знаний...")
    try:
        # Run ingestion in a separate thread/process to not block the bot
        # For simplicity, we call it directly but it might block if large
        # Better to run in executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, ingest.ingest_documents)
        await status_msg.edit_text("✅ База знаний успешно обновлена.")
    except Exception as e:
        logging.error(f"Refresh error: {e}")
        await status_msg.edit_text(f"❌ Ошибка обновления: {e}")

@dp.message(F.text)
async def handle_message(message: types.Message):
    user_query = message.text
    logging.info(f"Received query: {user_query}")
    
    status_msg = await message.answer("⚖️ Анализирую судебную практику...")
    
    try:
        # 1. Embed query
        query_embedding = embedding_model.encode(user_query).tolist()
        
        # 2. Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=7
        )
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        context_parts = []
        for i, doc in enumerate(documents):
            meta = metadatas[i]
            filename = meta.get('filename', 'Unknown')
            case_number = extract_case_number(filename)
            
            # Smart prefixing: don't double-add "Дело" if it's already there
            if case_number.lower().lstrip().startswith("дело") or case_number.lower().lstrip().startswith("case"):
                source_label = case_number
            else:
                source_label = f"Дело №{case_number}"
            
            # Format each chunk with explicit Source ID
            chunk_text = f"SOURCE_ID: [{source_label}]\nCONTENT: {doc}"
            context_parts.append(chunk_text)
            
        context_str = "\n\n---\n\n".join(context_parts)
        
        if not context_str:
            context_str = "В базе знаний нет релевантных документов."
        
        # 3. Construct Prompt
        full_prompt = f"""
        КОНТЕКСТ:
        {context_str}
        
        ВОПРОС ПОЛЬЗОВАТЕЛЯ:
        {user_query}
        """
        
        # 4. Call RouterAI
        async with ChatActionSender(bot=bot, chat_id=message.chat.id, action="typing"):
            response = await router_client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
        ai_reply = response.choices[0].message.content
        
        # Delete status message and send reply
        await status_msg.delete()
        await message.answer(ai_reply)
        
    except Exception as e:
        logging.error(f"Error handling message: {e}")
        await status_msg.edit_text("⚠️ Произошла ошибка при обработке запроса. Пожалуйста, попробуйте позже.")

async def main():
    logging.info("Bot started")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped")
    except Exception as e:
        print(f"Critical error: {e}")
        logging.exception("Critical error")
