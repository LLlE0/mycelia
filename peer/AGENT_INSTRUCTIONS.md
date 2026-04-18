# Инструкция для AI-агента: Исправление передачи батчей между узлами

## Текущая проблема
- PREP node создал 100 батчей
- PROC node не может получить батчи за 30 секунд
- WebSocket отваливается по ping/pong timeout

## Корневая причина
PROC node запрашивает батчи у PREP, но:
1. PREP не имеет прямого HTTP сервера для раздачи батчей
2. batch_sources не содержит IP:port PREP узла
3. Даже если бы содержал - PREP не слушает на этом порту

## Что нужно исправить

### 1. PREP узел должен запускать HTTP сервер для батчей
В peer/participant.py или peer/prep.py:
- При получении роли PREP запустить HTTP сервер на порту (например 11130)
- Сервер должен отвечать на GET /batch/<job_id>/<batch_num>
- Вернуть JSON с input_ids и attention_mask

### 2. Coordinator должен включать IP:port в batch_sources
В coordinator/server.go при отправке task_train:
- batch_sources должен быть: {batch_num: {name, ip, port}}
- Сейчас он просто {batch_num: node_name}

### 3. PREP должен сообщить свой порт при регистрации
В participant.py при регистрации:
- После получения роли PREP, запустить сервер
- Сообщить порт координатору через API

### 4. PROC должен запрашивать батчи напрямую
В peer/proc.py:
- Использовать batch_sources[batch_num].ip и batch_sources[batch_num].port
- Запросить GET http://ip:port/batch/job_id/batch_num
- Сохранить полученные данные локально

## Файлы для модификации

### peer/participant.py
- Добавить запуск HTTP сервера при роли PREP
- Добавить отправку порта координатору

### peer/proc.py  
- Изменить логику получения батчей на прямой HTTP

### coordinator/server.go
- Изменить формат batch_sources
- Использовать IP:port из participant record

### coordinator/database.go  
- Добавить метод UpdateParticipant если нужен

## Ожидаемый результат
После запуска задачи:
1. PREP запускает HTTP сервер на порту 11130
2. Coordinator знает где найти батчи (ip:port)
3. PROC запрашивает батчи напрямую через HTTP
4. WebSocket остается живым во время обучения
