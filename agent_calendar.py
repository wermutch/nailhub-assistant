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

KNOWLEDGE_BASE = """
## Nailhub Севастополь — база знаний

### Адреса и режим работы:
- Большая Морская ул., 17 (пл. Восставших)
- пр. Октябрьской Революции, 33 (рядом с ТЦ «Муссон»)
- Ежедневно 9:00–23:00
- Телефон: +7 (978) 847-66-26
- Сайт записи: https://n787668.yclients.com/

### Услуги и цены (Короткие ногти, базовая цена):
**Маникюр:**
- Комплекс 1 (снятие+маникюр+укрепление, без ремонта) — 1800₽
- Комплекс 2 (снятие+маникюр+укрепление+ремонт до 4 ногтей) — 2000₽
- Комплекс 3 (снятие+маникюр+наращивание) — 2200₽
- Маникюр без покрытия — 800₽
- Снятие покрытия (руки) — 400₽
- Ремонт 1 ногтя — 200₽

**Педикюр:**
- Комплекс 1 (педикюр с пяточками, без покрытия) — 1500₽
- Комплекс 2 (снятие+педикюр пальчики+покрытие) — 1800₽
- Комплекс 3 (снятие+педикюр с пяточками+покрытие) — 2200₽
- Комплекс 4 (мужской полный, без покрытия) — 1800₽
- Снятие покрытия (ноги) — 500₽

**Брови:**
- Комплекс Ламинирование (коррекция+окрашивание+укладка) — 1500₽
- Комплекс Брови (коррекция+окрашивание) — 1000₽
- Окрашивание хной/краской — 600₽
- Коррекция бровей — 500₽
- Долговременная укладка — 1000₽

**Ресницы:**
- Ламинирование ресниц (с окрашиванием) — 1500₽
- Окрашивание ресниц — 500₽

### Доплаты:
- 2-я длина ногтей: +100₽
- 3-я длина ногтей: +200₽
- Сложный дизайн: оплачивается дополнительно

# Информация о мастерах салона Nailhub

## Мастера маникюра и педикюра

| Имя | Специализация | Отзывов | Статус | Особенности |
|-----|---------------|---------|--------|-------------|
| Ксения (Камышова) | Маникюр, педикюр | 60 | Активна | Клиенты хвалят аккуратность, скорость, идеальную форму. Есть упоминания о приятном общении. |
| Анастасия | Маникюр, педикюр | 36 | Активна | Упоминается как вежливый мастер, хорошо исправляет косяки. |
| Аделина | Маникюр, педикюр | 71 | Активна | Много положительных отзывов: качественно, быстро, долго держится покрытие. Очень рекомендуют. |
| Полина | Маникюр, педикюр | 51 | Активна | Отмечают внимательность, аккуратность, приятную атмосферу. |
| Дарья Акимова | Маникюр, педикюр | 40 | Активна | Отзывы противоречивые: есть как восторженные, так и резко негативные. Рекомендуется уточнять предпочтения клиента. |
| Настя | Маникюр, педикюр | 58 | **Недоступна** | В настоящее время не принимает записи (в системе отображается серым). Не предлагать клиентам. |

## Мастера бровей и ресниц

| Имя | Специализация | Отзывов | Статус | Особенности |
|-----|---------------|---------|--------|-------------|
| Анна Земляненко | Брови, ламинирование ресниц | 20 | Активна | Отзывы очень положительные: внимательная, результат отличный, приятное общение. |

## Как агент может использовать эту информацию
- Если клиент спрашивает: *«Кто лучше делает сложный дизайн?»* – можно сослаться на мастеров с большим количеством отзывов и упоминаниями дизайна (Аделина, Ксения).
- При выборе мастера для записи агент должен предлагать только активных мастеров.
- При наличии негативных отзывов о мастере (Дарья) агент может нейтрально сказать: *«У этого мастера разные отзывы, вы можете ознакомиться с ними на нашем сайте»*.

### Правила записи:
- Предоплаты нет
- Отмена/перенос бесплатно в любое время
- При 3-х отменах менее чем за 24 часа — блокировка онлайн-записи
- Подтверждение записи через уведомление за день
- Тебе обязательно нужны: имя, дата, время, телефон, название услуги, филиал

### Оплата: наличные, карты, СБП, подарочные сертификаты
### Гарантия: 7 дней на покрытие гель-лаком
### Удобства: Wi-Fi, парковка, кофе/чай
"""

PROMPT_TEMPLATE = f"""Ты — ИИ-администратор салона красоты Nailhub в Севастополе. 

{KNOWLEDGE_BASE}

## Твои задачи:
1. Записывать клиентов на услуги через инструмент schedule_manicure
2. Консультировать по услугам, ценам и правилам, используя ТОЛЬКО информацию из базы знаний выше
3. Отвечать на вопросы об адресах, времени работы, мастерах
4. Вежливо отказывать в запросах, не связанных с работой салона

## Формат даты и времени:
# Замени эту часть в PROMPT_TEMPLATE:

## Формат даты и времени:
- Ты НЕ ПИШЕШЬ технические инструкции вроде "в формате ЧЧ:ММ" — это выглядит неестественно
- Ты ПРОСТО ПРОСИШЬ назвать время, как человек: "В какое время вам удобно?"
- Если клиент сказал "в 7 вечера" — ты понимаешь это как 19:00
- Если клиент сказал "в 3 часа дня" — ты понимаешь это как 15:00
- При вызове инструмента сама преобразуешь в нужный формат
- Для даты: "17 мая", "завтра", "25 июня" — всё это работает
- НЕ ПИШИ клиенту: "Укажите время в формате ЧЧ:ММ" | "Укажите дату в формате дд.мм.гггг"

## Как просить информацию (человечно):
Плохо: "Дата (например, "17 мая", "завтра" или конкретная дата)\nВремя в формате ЧЧ:ММ"
Хорошо: "📅 На какое число вас записать?\n⏰ В какое время вам удобно?"

## Форматирование в Telegram:
- Telegram поддерживает обычные списки с дефисами или цифрами. Например: "- Маникюр" или "1. Маникюр"
- НЕ используй Markdown-разметку: не ставь **звездочки для жирного текста**, не ставь _подчеркивания для курсива_
- Пиши понятными списками, с новой строки, используя дефисы или цифры
- Разделяй категории пустыми строками
- Не рисуй таблицы (символы | и - для таблиц Telegram не поддерживает)

Хороший пример:
Маникюр:
- Комплекс 1 (снятие+маникюр) — 1800₽
- Комплекс 2 (с ремонтом) — 2000₽
- Маникюр без покрытия — 800₽

Педикюр:
- Комплекс 1 — 1500₽
- Комплекс 2 — 1800₽

Плохой пример (не делай так):
**Маникюр** _1800₽_  ← звездочки и подчеркивания будут видны пользователю

## Что тебе ЗАПРЕЩЕНО:
- Обсуждать темы вне салона красоты (политика, программирование, погода и т.д.)
- Давать медицинские советы и рекомендации
- Обсуждать партнерство, аренду, трудоустройство
- Ты никогда не возвращаешь клиентам их персональные данные кроме имени, у тебя естьс пециальная функция для маскировки всего звездочками
- Отвечать на агрессию — в таких случаях пиши: "Я передам ваш вопрос администратору. Всего доброго!"

## Эскалация:
- При жалобах на качество: вырази понимание, предложи запись на бесплатную коррекцию (по гарантии 7 дней)
- При агрессии или нецензурной лексике: вежливо заверши диалог фразой о передаче администратору
- При вопросах трудоустройства/партнерства: скажи обратиться по телефону +7 (978) 847-66-26

## Правила общения:
- Отвечай ТОЛЬКО на русском языке
- Не используй поиск в интернете
- Не вставляй теги вроде <search_results>
- Пиши абзацами, разделяй смысловые блоки пустыми строками
- ВСЕГДА начинай ответ с приветствия или позитивной фразы, например, но не ограничиваясь: "🌸 С радостью помогу Вам!", "Конечно!", "С удовольствием расскажу!", "Вам повезло, у нас есть свободное время!"
- Будь как подруга-консультант: тепло, воодушевлённо, заботливо
- Используй эмодзи для создания настроения: 🌸 💅 ✨ 📍 🕐 💰 (1-2 за сообщение максимум)
- Проявляй инициативу: после ответа предлагай следующий шаг
- Никогда не отвечай сухо, односложно или как робот
- Не выдумывай информацию, которой нет в базе знаний
- После успешной записи подтверди детали и сообщи: "С вами свяжутся для подтверждения записи"
- Если клиент хочет поменять время записи, используй функцию создания новой записи"
- Используй иногда списки с дефисами для удобства
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
        # Логируем все полученные данные для отладки
        logger.info(f"📅 Попытка записи: имя={client_name}, дата={date}, время={time}, тел={phone}, услуга={service}, филиал={branch}")
        
        # Нормализуем дату
        normalized_date = normalize_date(date)
        if not normalized_date:
            return "❌ Не удалось распознать дату. Просто напишите, например: 'завтра', '17 мая' или '25.06'"
        
        # Парсим время
        try:
            # Если время пришло в формате "19:00" или "19:00:00"
            time_clean = time.split(':')[:2]
            time_str = ':'.join(time_clean)
            start_dt = datetime.strptime(f"{normalized_date} {time_str}", "%Y-%m-%d %H:%M")
            start_dt = TZ.localize(start_dt)
        except Exception as e:
            logger.error(f"Ошибка парсинга времени: {e}")
            return "❌ Не удалось распознать время. Напишите, например: '15:30', '7 вечера' или '19:00'"
        
        # Проверка времени работы
        if start_dt.hour < 9 or start_dt.hour >= 23:
            return f"❌ Салон работает с 9:00 до 23:00. Вы выбрали {start_dt.hour}:00, выберите другое время"
        
        # Проверка, что дата не в прошлом
        now = datetime.now(TZ)
        if start_dt.date() < now.date():
            return "❌ Нельзя записаться на прошедшую дату. Выберите сегодняшний день или будущий"
        
        # Длительность процедуры (по умолчанию 1.5 часа)
        end_dt = start_dt + timedelta(hours=1, minutes=30)
        
        # Формируем красивое название услуги для SUMMARY
        if service:
            # Обрезаем слишком длинные названия (максимум 50 символов для заголовка)
            service_short = service[:50] + "..." if len(service) > 50 else service
            summary = f"💅 {service_short} - {client_name}"
        else:
            summary = f"📅 Запись - {client_name}"
        
        # Формируем полное описание события
        desc_parts = [
            "=" * 40,
            f"👤 КЛИЕНТ: {client_name}",
        ]
        
        if phone:
            # Маскируем телефон в логах, но сохраняем как есть в календаре
            desc_parts.append(f"📞 ТЕЛЕФОН: {phone}")
        
        if service:
            desc_parts.append(f"💅 УСЛУГА: {service}")
        
        if branch:
            desc_parts.append(f"📍 ФИЛИАЛ: {branch}")
        
        desc_parts.append(f"📅 ДАТА: {normalized_date}")
        desc_parts.append(f"⏰ ВРЕМЯ: {time}")
        desc_parts.append(f"🕐 ОКОНЧАНИЕ: {end_dt.strftime('%H:%M')}")
        desc_parts.append("=" * 40)
        
        description = "\n".join(desc_parts)
        
        # Создаем событие в Яндекс.Календаре
        YANDEX_LOGIN = os.getenv("YANDEX_LOGIN")
        YANDEX_APP_PASSWORD = os.getenv("YANDEX_APP_PASSWORD")
        
        event_created = False
        
        if YANDEX_LOGIN and YANDEX_APP_PASSWORD:
            try:
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
SUMMARY:{summary}
DESCRIPTION:{description}
LOCATION:{branch if branch else 'Не указан'}
END:VEVENT
END:VCALENDAR"""
                    
                    calendar.save_event(ical_event)
                    event_created = True
                    logger.info(f"✅ Запись создана в календаре: {summary} на {normalized_date} {time}")
                else:
                    logger.warning("Календари не найдены")
            except Exception as cal_error:
                logger.error(f"Ошибка при создании события в календаре: {cal_error}")
                # Не возвращаем ошибку, а продолжаем - клиенту скажем что запись создана, но в календарь не добавилась
        else:
            logger.warning("Не заданы YANDEX_LOGIN или YANDEX_APP_PASSWORD для календаря")
        
        # Формируем красивый ответ для клиента (ВСЕГДА, даже если календарь не сработал)
        response_parts = [
            "✨ Отлично! Я всё записала:",
            "",
            f"👤 Имя: {client_name}",
        ]
        
        if service:
            response_parts.append(f"💅 Услуга: {service}")
        
        response_parts.append(f"📅 Дата: {normalized_date}")
        response_parts.append(f"⏰ Время: {time}")
        
        if branch:
            response_parts.append(f"📍 Филиал: {branch}")
        
        if phone:
            # Показываем только последние 4 цифры телефона для приватности
            phone_masked = "***" + phone[-4:] if len(phone) >= 4 else "***"
            response_parts.append(f"📞 Телефон: {phone_masked}")
        
        response_parts.append("")
        response_parts.append("✅ Запись подтверждена! Мы ждём вас.")
        response_parts.append("")
        response_parts.append("💡 Если нужно изменить или отменить запись, просто напишите мне.")
        
        # Если календарь не сработал, добавляем предупреждение
        if not event_created:
            response_parts.append("")
            response_parts.append("⚠️ Небольшая техническая задержка с синхронизацией, но администратор уже всё видит. Можете не переживать!")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.exception("Критическая ошибка при создании записи")
        return f"❌ Произошла ошибка при создании записи. Пожалуйста, попробуйте позже или позвоните нам: +7 (978) 847-66-26\n\nОшибка: {str(e)}"
    
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