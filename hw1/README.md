# RecSys Домашнее задание 1

Баклыкова Алина, Б05-322 
Датасет: Amazon Reviews 2018 - Digital Music (5-core)  
Ссылка: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Digital_Music_5.json.gz

Используется подвыборка из 5 000 самых популярных товаров.

## Задача

Реализация пяти моделей с нуля и сравнение персональных top-10 рекомендаций:

1. Item-based collaborative filtering
2. Implicit ALS
3. EASE
4. Factorization Machine с метапризнаками
5. Implicit ALS с собственным целевым сигналом

## Структура

- `experiments.ipynb` - основной ноутбук с выполненными ячейками, графиками и таблицами результатов
- `models/` - реализация каждой из пяти моделей
- `requirements.txt` - зависимости
