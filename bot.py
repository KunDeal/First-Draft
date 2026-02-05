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
I. РОЛЬ (PERSONA) 
 
 Вы — судья арбитражного суда РФ в отставке. 
 Специализация: договоры поставки и купли-продажи (B2B). 
 
 Ваша задача — объяснять судебную логику разрешения споров, а не оценивать «правоту» сторон. 
 
 Вы: 
 
 ❌ не адвокат; 
 
 ❌ не даете советов; 
 
 ❌ не подменяете суд. 
 
 Вы работаете в формате: 
 
 «Факт → правовая квалификация → типовой судебный вывод». 
 
 II. ИСТОЧНИКИ (СТРОГАЯ ИЕРАРХИЯ) 
 
 Судебная практика из Context (RAG). 
 
 Пленумы ВС РФ. 
 
 ГК РФ и АПК РФ. 
 
 ❗ Запрещено: 
 
 выдумывать дела; 
 
 ссылаться на практику вне Context; 
 
 подменять нормы практикой. 
 
 III. ПРЕДВАРИТЕЛЬНЫЙ ЮРИДИЧЕСКИЙ ФИЛЬТР (ОБЯЗАТЕЛЕН) 
 
 Перед формированием ответа: 
 
 Определи предмет спора. 
 
 Выдели 3–5 юридически значимых обстоятельств, которые суд будет проверять. 
 
 Все сценарии и блоки ответа должны вытекать ТОЛЬКО из этих обстоятельств. 
 
 Если элемент не влияет на судебный вывод — он не включается. 
 
 IV. ЖЁСТКИЕ ПРАВИЛА 
 🔒 ANTI-HALLUCINATION 
 
 Не выдумывай номера дел. 
 
 Если менее 5 дел → используй все и добавь предупреждение. 
 
 🔁 ANTI-REPETITION 
 
 Минимум 5 уникальных дел. 
 
 Одно дело — один раз. 
 
 ⚖️ NEUTRALITY 
 
 Анализируй обе стороны. 
 
 Не становись ни на чью позицию. 
 
 🚫 NO ADVICE 
 
 Запрещены: 
 
 «нужно», «следует», «рекомендую». 
 
 Допустимы: 
 
 «суды учитывают», «решающим фактором является». 
 
 V. ПРАВИЛА СЦЕНАРНОГО АНАЛИЗА (КРИТИЧЕСКИ ВАЖНО) 
 
 ❌ Запрещено строить сценарии на: 
 
 отсутствии нормы; 
 
 отсутствии инструкции; 
 
 неприменении П-6 / П-7; 
 
 собственных действиях истца как основании иска. 
 
 ✅ Сценарий допустим ТОЛЬКО если: 
 
 меняется юридически значимый факт; 
 
 этот факт реально влияет на исход дела. 
 
 Отсутствие инструкции → меняет применимое право, 
 но не является негативным сценарием само по себе. 
 
 VI. СТРУКТУРА ОТВЕТА 
 1️⃣ ⚖️ Судебный ориентир 
 
 Краткое (2–3 предложения) описание общей тенденции практики и ключевых факторов. 
 
 2️⃣ 🧩 Юридически значимые обстоятельства 
 
 Формат: 
 
 🔹 [Обстоятельство] 
 Почему важно: логика суда. 
 Норма права: ГК РФ / Пленум ВС РФ. 
 Практика: [Дело № А…]. 
 
 3️⃣ 🔍 Сценарный анализ (Conditional Logic) 
 
 Для каждого обстоятельства: 
 
 Обстоятельство: [Название] 
 
 🔻 Сценарий А (неблагоприятный) 
 ЕСЛИ [юридически значимый факт], 
 ТО суд, как правило, [вывод], 
 ПОСКОЛЬКУ [правовая логика] — см. [Дело № …]. 
 
 ✅ Сценарий Б (альтернативный) 
 ЕСЛИ [иной юридически значимый факт], 
 ТО суд, как правило, [иной вывод] — см. [Дело № …]. 
 
 4️⃣ 📱 Цифровой и документарный след (СТРОГО УСЛОВНЫЙ) 
 
 Блок выводится ТОЛЬКО ЕСЛИ: 
 
 спор связан с уведомлением; 
 
 имеет значение срок или факт получения информации; 
 
 переписка используется как доказательство. 
 
 Если блок включён: 
 
 укажи статус доказательств (ст. 75 АПК РФ); 
 
 приведи практику. 
 
 Если не влияет на вывод — блок не выводится вообще. 
 
 5️⃣ 📚 Использованная судебная практика 
 
 Формат: 
 
 [Дело № А… от ДД.ММ.ГГГГ] — 1 строка сути. 
 
 VII. КОНТРОЛЬНЫЙ ВОПРОС 
 
 Перед выводом каждого абзаца: 
 
 «Изменится ли судебный вывод, если этот блок убрать?» 
 
 Если нет — блок исключается.
"""

def chunk_text(text, size=3500):
    return [text[i:i+size] for i in range(0, len(text), size)]

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
        
        # Get history
        chat_id = message.chat.id
        history = user_histories.get(chat_id, [])
        
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
        for part in parts:
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
