# Запуск своего ОСВЕД

## 1. Что подготовить

1. Docker с Compose и компьютер или VPS, который будет включён для работы бота.
2. `TELEGRAM_BOT_TOKEN` — создай отдельного бота у [BotFather](https://t.me/BotFather).
3. `ALLOWED_USER_ID` — числовой ID твоего личного Telegram-аккаунта, не @username и не ID канала.
4. `OPENROUTER_API_KEY` — [ключ OpenRouter](https://openrouter.ai/settings/keys) и баланс.
5. `MONID_API_KEY` — [ключ Monid](https://app.monid.ai/access/api-keys) и баланс.

Для числового ID можно открыть официальный пример `getUpdates` в
[документации Bot API](https://core.telegram.org/bots/api#getupdates): после сообщения
новому боту ID владельца находится в `message.from.id`. Делай запрос локально;
не публикуй URL с бот-токеном и не запускай polling одновременно со своим сервером.

## 2. Скачать и настроить

```sh
git clone https://github.com/SlavaXAi/osved.git
cd osved
cp .env.example .env
```

Открой `.env`, заполни четыре значения из списка. Не коммить этот файл.
В `config/interests.yaml` замени `topics`, `regions_of_interest`, `goals` и
предпочтительные типы событий на свои. Стартовые подписки пустые.

```sh
docker compose build
docker compose run --rm newsfeed osved doctor
docker compose up -d
docker compose logs --tail 50 newsfeed
```

Напиши своему боту `/start`. Работает один владелец, указанный в `.env`.

## 3. Telegram и YouTube

**Telegram:** пришли публичную ссылку `t.me/name`, `@name` или пересланный пост.
Проверь название и нажми «Добавить в радар». Управление — `/tgchannels`.
Закрытые каналы, группы и личные аккаунты не поддерживаются этим способом.
Новый канал начинает с последних суток и дальше проходит общий отбор событий.

**YouTube:** пришли ссылку на канал или `/channel ссылка`. Управление — `/channels`.
Новые выпуски отслеживаются по RSS. Ссылка на отдельное видео запускает краткую
сводку; кнопка «Длинный отчёт · PDF» или `/report ссылка` — полный отчёт.
Транскрипты извлекаются только по запросу. Google-вход и браузер не нужны.

Новости и поиск используют модели из `.env`. Сводка видео и полный PDF сейчас
используют [Gemini 3.8 Flash через OpenRouter](https://openrouter.ai/google/gemini-3.8-flash),
зафиксированный в `app/video/report_format.py`; доступ к этой модели нужен на твоём ключе.

## 4. Подключить X/Twitter

Публичные авторы читаются через Monid. Сначала скопируй пример:

```sh
cp examples/x-authors.yaml config/x-authors.yaml
```

Замени пример своими авторами. Один элемент выглядит так:

```yaml
version: 1
authors:
  - handle: karpathy
    name: Мой источник
    topics: [ai, technology]
    initial_tier: active
    enabled: true
    weekly_persona: false
```

`handle` — имя без `@`. `initial_tier` задаёт частоту: `stream` — два раза в сутки,
`active` — раз в два дня, `sparse` — раз в неделю. Значения действуют вместе с
расписанием, доступностью API и лимитами.

Для поиска по теме:

```sh
cp examples/sensory-sources.yaml config/sensory-sources.yaml
```

Измени `locator` на свой поисковый запрос и `topics` на нужные темы. Можно добавить
несколько элементов с уникальными `id`. После изменения проверь `/sources` и `/radar`.
X-посты служат сигналами для исследования; модель не обязана превратить каждый твит в новость.
Twitter API token, cookies и вход в личный аккаунт для этих режимов не нужны.

## 5. Включить фоновую работу

```sh
docker compose exec newsfeed osved schedule-enable --dry-run
docker compose exec newsfeed osved schedule-status
docker compose exec newsfeed osved schedule-enable --live
```

Последняя команда включает платные API-операции по расписанию. В стартовой
конфигурации часовой пояс — `Europe/Moscow`, сбор — 10:00–20:00 с интервалом два
часа. `SCHEDULE_TIMEZONE`, `RADAR_TIMES`, `RESEARCH_TIMES` и `DELIVERY_TIMES`
можно изменить в `.env`; время задаётся массивом строк `HH:MM`.
Для круглосуточного сбора нужно также согласовать интервалы и дневные лимиты источников.

Тематический поиск задаётся в `config/search-agents.yaml`; его выключение —
`SCHEDULED_SEARCH_ENABLED=false`. Отключение всех каналов отдельно не отключает web search.

## 6. Проверка, остановка и обновление

```sh
docker compose exec newsfeed osved sensors-status
docker compose exec newsfeed osved schedule-status
docker compose exec newsfeed osved details-status
docker compose exec newsfeed osved schedule-disable
docker compose stop
```

`schedule-disable` останавливает новые фоновые проходы; текущая работа может закончиться.
Кнопки сводок и подробного разбора остаются отдельными запросами. `stop` останавливает сервис.

База и результаты хранятся в Docker volume `data`, дополнительные экспорты — в `exports`.
Перед обновлением сделай `docker compose exec newsfeed osved backup-db`.
Чтобы сохранить копию на компьютере, выполни
`docker compose cp newsfeed:/app/data/backups ./osved-backups`.
После `git pull` используй `docker compose up -d --build`. После изменения `.env` —
`docker compose up -d --force-recreate`.

Не используй `docker compose down -v`, если хочешь сохранить историю: флаг `-v` удаляет volumes.
Для другого владельца используй отдельный экземпляр и пустую базу.
Не запускай два polling-процесса с одним бот-токеном. Для поиска причины редкой
выдачи смотри сбор, отбор событий и исследование отдельно: работающий процесс
не доказывает полноту радара.
