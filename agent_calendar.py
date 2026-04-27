import os
import logging
import time
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import create_agent
import caldav
from datetime import datetime, timedelta
import uuid
from typing import Dict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from dateparser import parse as dateparse
import pytz

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("manicure_bot")

# Часовой пояс салона
TZ = pytz.timezone("Europe/Moscow")

PROMPT_TEMPLATE = f"""Ты — дружелюбный и заботливый ИИ-администратор салона красоты Nailhub в Севастополе. Твоя задача — помогать клиентам с теплотой и энтузиазмом.

{KNOWLEDGE_BASE}

## СТИЛЬ ОБЩЕНИЯ (ЭТО ВАЖНО!):
- ВСЕГДА начинай ответ с приветствия или позитивной фразы: "🌸 С радостью!", "Конечно!", "С удовольствием расскажу!", "Вам повезло, у нас есть свободное время!"
- Будь как подруга-консультант: тепло, воодушевлённо, заботливо
- Используй эмодзи для создания настроения:
- Проявляй инициативу: после ответа предлагай следующий шаг
- Никогда не отвечай сухо, односложно или как робот

## ПРИМЕРЫ ПРАВИЛЬНЫХ ОТВЕТОВ:
Клиент: "что у вас по ценам?"
Правильно: "🌸 С удовольствием расскажу! У нас демократичные цены и всегда качественные материалы. Вот наш прайс:

Маникюр:
- Классический с покрытием — 1800₽
- Без покрытия — 800₽
- Снятие — 400₽

Педикюр от 1500₽, брови от 500₽. Какая услуга вас интересует? 💅"

Неправильно: "Маникюр 1800 руб, педикюр 1500 руб" (сухо и недружелюбно)

Клиент: "хочу записаться"
Правильно: "🌸 Отлично! Буду рада вас записать! Для записи мне понадобятся: ваше имя, желаемая дата и время, услуга и филиал. Назовите их, пожалуйста. 📍"

## Твои задачи:
1. Записывать клиентов на услуги через инструмент schedule_manicure
2. Консультировать по услугам, ценам и правилам, используя ТОЛЬКО информацию из базы знаний

## Формат даты и времени:
- Принимай даты в естественном формате: "17 мая", "завтра", "послезавтра"
- В инструмент передавай дату в формате ГГГГ-ММ-ДД
- Время в формате ЧЧ:ММ

## Форматирование в Telegram:
- Используй списки с дефисами (-) для удобства
- НЕ используй Markdown-разметку (звездочки ** и подчеркивания _ будут видны)
- Используй эмодзи для дружелюбия: 🌸 💅 ✨ 📍 🕐 💰 ☎️

## Что ЗАПРЕЩЕНО:
- Обсуждать темы вне салона
- Давать медицинские советы
- Быть сухим, роботизированным, односложным
- Использовать Markdown (**жирный**, _курсив_)

## Правила:
- ВСЕГДА будь дружелюбной и приветливой
- Проявляй инициативу
- После записи подтверди детали и скажи: "С вами свяжутся для подтверждения ✨"
- При агрессии: "Я передам ваш вопрос администратору. Всего доброго!"
- При жалобах: предложи запись на бесплатную коррекцию по гарантии 7 дней
"""

# --- Инициализация LLM ---
provider = os.getenv("PROVIDER_NAME", "").upper()
api_key = os.getenv("API_KEY")
model = os.getenv("MODEL_NAME")

if provider == "OPENROUTER":
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=model or "openai/gpt-3.5-turbo",
        temperature=0.3,  # Снижено для большей точности
    )
else:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# --- Функция нормализации даты ---
def normalize_date(date_str: str) -> Optional[str]:
    """Преобразует дату в формате естественного языка в YYYY-MM-DD"""
    try:
        parsed = dateparse(
            date_str,
            languages=['ru'],
            settings={
                'PREFER_DATES_FROM': 'future',
                'DATE_ORDER': 'DMY',
                'PREFER_DAY_OF_MONTH': 'current',
            }
        )
        if parsed:
            return parsed.strftime("%Y-%m-%d")
    except:
        pass
    # Пробуем стандартные форматы
    formats = ["%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except:
            continue
    return None

# --- Инструменты ---
@tool
def schedule_manicure(client_name: str, date: str, time: str, phone: str = None, service: str = None, branch: str = None) -> str:
    """
    Записывает клиента в Яндекс.Календарь.
    Параметры:
    - client_name: имя клиента
    - date: дата в формате ГГГГ-ММ-ДД
    - time: время в формате ЧЧ:ММ
    - phone: телефон
    - service: название услуги
    - branch: филиал
    """
    try:
        normalized_date = normalize_date(date)
        if not normalized_date:
            return "❌ Не удалось распознать дату. Пожалуйста, укажите в формате 'день месяц' или '2026-05-17'"
        
        start_dt = datetime.strptime(f"{normalized_date} {time}", "%Y-%m-%d %H:%M")
        start_dt = TZ.localize(start_dt)
        
        # Проверка времени работы
        if start_dt.hour < 9 or start_dt.hour >= 23:
            return "❌ Салон работает с 9:00 до 23:00. Пожалуйста, выберите другое время."
        
        end_dt = start_dt + timedelta(hours=1, minutes=30)
        
        # Формируем описание события
        desc_parts = [f"Клиент: {client_name}"]
        if phone:
            desc_parts.append(f"Телефон: {phone}")
        if service:
            desc_parts.append(f"Услуга: {service}")
        if branch:
            desc_parts.append(f"Филиал: {branch}")
        
        # Создаем событие в календаре
        YANDEX_LOGIN = os.getenv("YANDEX_LOGIN")
        YANDEX_APP_PASSWORD = os.getenv("YANDEX_APP_PASSWORD")
        
        if YANDEX_LOGIN and YANDEX_APP_PASSWORD:
            client = caldav.DAVClient(
                url="https://caldav.yandex.ru",
                username=YANDEX_LOGIN,
                password=YANDEX_APP_PASSWORD
            )
            principal = client.principal()
            calendars = principal.calendars()
            if calendars:
                calendar = calendars[0]
                uid = str(uuid.uuid4())
                start_str = start_dt.strftime("%Y%m%dT%H%M%S")
                end_str = end_dt.strftime("%Y%m%dT%H%M%S")
                now_str = datetime.now(TZ).strftime("%Y%m%dT%H%M%S")
                
                ical_event = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Nailhub Bot//EN
BEGIN:VEVENT
UID:{uid}
DTSTAMP:{now_str}
DTSTART:{start_str}
DTEND:{end_str}
SUMMARY:Запись - {client_name}
DESCRIPTION:{chr(10).join(desc_parts)}
END:VEVENT
END:VCALENDAR"""
                calendar.save_event(ical_event)
                logger.info("Создана запись: %s на %s %s", client_name, normalized_date, time)
        
        # Формируем ответ
        response_parts = [
            f"✅ Запись создана!",
            f"Имя: {client_name}",
            f"Дата: {normalized_date}",
            f"Время: {time}",
        ]
        if service:
            response_parts.append(f"Услуга: {service}")
        if branch:
            response_parts.append(f"Филиал: {branch}")
        response_parts.append("С вами свяжутся для подтверждения записи.")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.exception("Ошибка создания записи")
        return "❌ Произошла ошибка при создании записи. Пожалуйста, попробуйте позже или позвоните нам: +7 (978) 847-66-26"

# --- Маскирование конфиденциальных данных ---
def mask_sensitive_data(text: str) -> str:
    """Маскирует email-адреса и номера телефонов"""
    if not text:
        return text
    
    # Маскируем email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    text = re.sub(email_pattern, lambda m: '*' * len(m.group()), text)
    
    # Маскируем телефоны
    phone = r'(\+7|8)[\s(]*\d{3}[\s)]*\d{3}[\s-]*\d{2}[\s-]*\d{2}'
    text = re.sub(phone, '***', text)
    
    return text

# --- Создание промпта и агента ---
prompt = ChatPromptTemplate.from_messages([
    ("system", PROMPT_TEMPLATE),
    MessagesPlaceholder(variable_name="messages"),
])

tools = [schedule_manicure]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=prompt.messages[0].prompt.template,
)

# --- Telegram бот ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MAX_TURNS_PER_CHAT = int(os.getenv("MAX_TURNS_PER_CHAT", "20"))
_memory_by_chat: Dict[int, List[BaseMessage]] = {}

def _get_history(chat_id: int) -> List[BaseMessage]:
    if chat_id not in _memory_by_chat:
        _memory_by_chat[chat_id] = []
    return _memory_by_chat[chat_id]

def _trim_history(history: List[BaseMessage]) -> None:
    max_messages = max(2, MAX_TURNS_PER_CHAT * 2)
    if len(history) > max_messages:
        del history[:-max_messages]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🌸 Добро пожаловать в Nailhub Севастополь!\n\n"
        "Я помогу вам:\n"
        "• Записаться на маникюр, педикюр, брови или ресницы\n"
        "• Узнать цены и адреса\n"
        "• Ответить на вопросы\n\n"
        "Напишите, что вас интересует!"
    )

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    
    MAX_LENGTH = 1000
    if len(text) > MAX_LENGTH:
        await update.message.reply_text("Сообщение слишком длинное. Пожалуйста, опишите ваш запрос короче.")
        return

    history = _get_history(chat_id)
    #masked_text = mask_sensitive_data(text)
    history.append(HumanMessage(content=text))
    _trim_history(history)

    try:
        t0 = time.perf_counter()
        result = agent.invoke({"messages": history})
        ai_text = result["messages"][-1].content
        masked_ai_text = mask_sensitive_data(ai_text)
        dt_ms = int((time.perf_counter() - t0) * 1000)
        logger.info("Ответ за %s мс, длина: %s символов", dt_ms, len(masked_ai_text))
    except Exception as e:
        logger.exception("Ошибка агента")
        await update.message.reply_text(
            "❌ Извините, произошла ошибка. Пожалуйста, позвоните нам: +7 (978) 847-66-26"
        )
        return

    history.append(AIMessage(content=masked_ai_text))
    _trim_history(history)
    await update.message.reply_text(masked_ai_text)

async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🎤 К сожалению, я пока не умею обрабатывать голосовые сообщения. "
        "Пожалуйста, напишите ваш вопрос текстом, и я с радостью помогу!"
    )

async def on_unsupported(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Я понимаю только текстовые сообщения. Пожалуйста, напишите ваш вопрос."
    )

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")
    
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Sticker.ALL | filters.VIDEO, on_unsupported))
    
    logger.info("Бот Nailhub запущен")
    app.run_polling()

if __name__ == "__main__":
    main()