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
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
LOGS_DIR = "logs"
MODEL_DIR = os.path.join("models", "e5-small")

# Setup logging
os.makedirs(LOGS_DIR, exist_ok=True)
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
    if os.path.isdir(MODEL_DIR):
        embedding_model = SentenceTransformer(MODEL_DIR)
    else:
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

# In-memory history storage
user_histories = {}
HISTORY_LIMIT = 6  # Keep last 3 exchanges

SYSTEM_PROMPT = """ 
 ТЫ — ЭКСПЕРТ-СУДЕБНИК (IP LITIGATOR). 
 Твоя специализация: споры по интеллектуальной собственности (авторское право, товарные знаки, патенты) в Суде по интеллектуальным правам (СИП) и Арбитражных судах РФ. 
 
 ТВОЯ ЦЕЛЬ: 
 Дать пользователю жесткий, реалистичный прогноз исхода дела и финансовой оценки, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном контексте судебных актов. Ты не цитируешь закон ради закона, ты ищешь прецеденты. 
 
 РЕЖИМЫ РАБОТЫ (Определи по запросу): 
 
 1. РЕЖИМ "ЗАЩИТА" (Пользователь — Ответчик, ему прилетела претензия): 
    - Ищи основания для СНИЖЕНИЯ компенсации (ст. 1301 ГК РФ, Постановление КС РФ 28-П). 
    - Ищи процессуальные дефекты истца (нет прав, злоупотребление, троллинг). 
    - Твой ответ должен быть "щитом": как заплатить минимум или не платить вовсе. 
 
 2. РЕЖИМ "НАПАДЕНИЕ" (Пользователь — Истец, у него украли): 
    - Оценивай реальность взыскания. Если он просит 5 млн за одно фото — охлади его пыл ссылками на практику, где дают 10к. 
    - Ищи доказательства, которые суды принимают (скриншоты, нотариус, веб-архив). 
    - Твой ответ должен быть "калькулятором ROI": стоит ли судиться. 
 
 СТРУКТУРА ОТВЕТА (Строго соблюдай): 
 
 1. 🎯 ВЕРДИКТ (TL;DR): 
    - Шанс на успех: [Высокий / 50 на 50 / Низкий]. 
    - Прогноз суммы: "Взыщут от X до Y рублей" (или "Откажут полностью"). 
 
 2. 💰 ФИНАНСОВЫЙ АНАЛИЗ: 
    - Объясни, почему такая сумма. Ссылайся на конкретные дела из контекста, где суд снизил или утвердил расчет. 
    - Пример: "Хотя истец требует 100к, в деле А56-... за аналогичное нарушение (1 фото) суд снизил сумму до 10к". 
 
 3. 🛡️/⚔️ СТРАТЕГИЯ (Аргументы): 
    - Список тезисов для иска или отзыва. 
    - Ссылка на "железобетонные" доказательства из контекста (например: "Суд принимает Web Archive как доказательство, см. дело А56-..."). 
 
 4. ⚖️ ПРЕЦЕДЕНТЫ (Ссылки): 
    - Список номеров дел из контекста, которые подтверждают твои слова. 
 
 СТРОГИЕ ОГРАНИЧЕНИЯ: 
 - ЗАПРЕЩЕНО выдумывать дела или факты. Если в контексте нет похожей ситуации, скажи: "В моей базе пока нет точного аналога, но исходя из общих тенденций...". 
 - ЗАПРЕЩЕНО лить воду ("важно отметить", "в соответствии с законодательством"). Пиши сухо, как в юридическом заключении (Legal Opinion). 
 - Приоритет отдавай Свежей практике (2024-2026). 
 """

def chunk_text(text, size=3500):
    return [text[i:i+size] for i in range(0, len(text), size)]

def extract_case_number(filename):
    """
    Extracts case number from filename using regex.
    Pattern: A\\d{2}-\\d{3,}/?\\d{2,4} (Case-insensitive, handling Cyrillic 'А' and Latin 'A').
    Example: "Дело №А40-12854-2013.docx" -> "А40-12854-2013"
    Fallback: clean filename without extension.
    """
    # Pattern to look for: A\\d{2}-\\d{3,}/?\\d{2,4}
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
    """Start command - clears history for a fresh start."""
    chat_id = message.chat.id
    if chat_id in user_histories:
        del user_histories[chat_id]
    
    welcome_text = (
        "Здравствуйте! Я ваш консультант по спорам в сфере поставки и купли-продажи по законодательству РФ.\n\n"
        "Для проведения анализа опишите вашу ситуацию текстом. Укажите:\n"
        "1. Суть спора\n"
        "2. Позиции сторон\n"
        "3. Ключевые обстоятельства\n\n"
        "Я работаю только с текстом вашего вопроса, документы прикреплять не нужно."
    )
    await message.answer(welcome_text)

@dp.message(Command("reset"))
async def cmd_reset(message: types.Message):
    """Clear conversation history."""
    chat_id = message.chat.id
    if chat_id in user_histories:
        del user_histories[chat_id]
    await message.answer("🧹 История диалога очищена. Я готов к новой теме.")

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

def get_after_response_keyboard():
    buttons = [
        [types.InlineKeyboardButton(text="✍️ Уточнить", callback_data="clarify")],
        [types.InlineKeyboardButton(text="🗑️ Завершить", callback_data="reset")]
    ]
    keyboard = types.InlineKeyboardMarkup(inline_keyboard=buttons)
    return keyboard


@dp.callback_query(F.data == "clarify")
async def process_clarify(callback: types.CallbackQuery):
    await callback.message.answer("Пожалуйста, напишите ниже дополнительные обстоятельства или уточнения. Я учту их при следующем ответе.")
    await callback.answer()

@dp.callback_query(F.data == "reset")
async def process_reset_callback(callback: types.CallbackQuery):
    """Callback handler to clear conversation history."""
    chat_id = callback.message.chat.id
    logging.info(f"Reset requested for chat_id: {chat_id}")
    if chat_id in user_histories:
        del user_histories[chat_id]
        logging.info(f"History deleted for chat_id: {chat_id}")
    else:
        logging.info(f"No history found to delete for chat_id: {chat_id}")
    
    # Double check deletion
    if chat_id in user_histories:
        logging.error(f"FAILED to delete history for chat_id: {chat_id}")
        user_histories[chat_id] = [] # Force empty

    await callback.message.answer("🧹 История диалога очищена. Я готов к новой теме.")
    await callback.answer()

@dp.message(F.text)
async def handle_message(message: types.Message):
    user_query = message.text
    chat_id = message.chat.id
    logging.info(f"Received query: {user_query} from chat_id: {chat_id}")

    # Greeting check
    greetings = ["привет", "здравствуйте", "добрый день", "hello", "hi", "start"]
    if user_query.lower().strip() in greetings or len(user_query.strip()) < 4:
        welcome_text = (
            "Здравствуйте! Я ваш консультант по спорам в сфере поставки и купли-продажи по законодательству РФ.\n\n"
            "Для проведения анализа опишите вашу ситуацию текстом. Укажите:\n"
            "1. Суть спора\n"
            "2. Позиции сторон\n"
            "3. Ключевые обстоятельства\n\n"
            "Я работаю только с текстом вашего вопроса, документы прикреплять не нужно."
        )
        await message.answer(welcome_text)
        return

    # Get history
    history = user_histories.get(chat_id, [])
    logging.info(f"Current history length for chat_id {chat_id}: {len(history)}")
    
    status_msg = await message.answer("Думаю...")
    
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
            context_chunk = f"SOURCE_ID: [{source_label}]\nCONTENT: {doc}"
            context_parts.append(context_chunk)
            
        context_str = "\n\n---\n\n".join(context_parts)
        
        if not context_str:
            context_str = "В базе знаний нет релевантных документов."
        
        # 3. Construct Prompt
        full_prompt = f"""
        КОНТЕКСТ (найденные документы):
        {context_str}
        
        ТЕКУЩИЙ ВОПРОС:
        {user_query}
        """
        
        # Get history (retrieved at start)
        
        # Construct messages list
        messages_payload = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": full_prompt}]

        # 4. Call RouterAI
        async with ChatActionSender(bot=bot, chat_id=message.chat.id, action="typing"):
            response = await router_client.chat.completions.create(
                model="google/gemini-3-flash-preview",
                messages=messages_payload,
                temperature=0.2,
                max_tokens=2000
            )
            
        ai_reply = response.choices[0].message.content
        
        # Update history
        # Store full prompt to keep context for future turns
        history.append({"role": "user", "content": full_prompt})
        history.append({"role": "assistant", "content": ai_reply})
        
        # Trim history
        if len(history) > HISTORY_LIMIT:
            history = history[-HISTORY_LIMIT:]
        
        user_histories[chat_id] = history
        parts = chunk_text(ai_reply, 3500)
        await status_msg.delete()
        
        # Send chunks, attach keyboard to the last one
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                await message.answer(part, reply_markup=get_after_response_keyboard())
            else:
                await message.answer(part)
        
    except Exception as e:
        logging.exception(f"Error handling message: {e}")
        try:
            await status_msg.edit_text("⚠️ Произошла ошибка при обработке запроса. Пожалуйста, попробуйте позже.")
        except Exception:
            await message.answer("⚠️ Произошла ошибка при обработке запроса. Пожалуйста, попробуйте позже.")

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
