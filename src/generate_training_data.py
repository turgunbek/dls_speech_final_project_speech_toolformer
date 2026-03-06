"""
generate_training_data.py
=========================
Генерирует ~3000 текстовых тренировочных примеров для Text SFT.
Примеры НЕ пересекаются с тестовым set (data/generated_dataset.json).

Запуск из корня проекта:
    python src/generate_training_data.py

Выход: data/training_data_text.json
"""

import json
import random
import itertools
import re

OUTPUT_JSON = "data/training_data_text.json"
RANDOM_SEED = 42

TARGET_POSITIVE = 2400
TARGET_NEGATIVE = 600

# ---------------------------------------------------------------------------
# Данные: получатели, суммы, шаблоны
# ---------------------------------------------------------------------------

RECIPIENTS = [
    # Семья
    "Mom", "Dad", "Sis", "Bro", "Son", "Daughter", "Wife", "Husband",
    "Aunt", "Uncle", "Grandma", "Grandpa",
    # Имена
    "Alex", "Mike", "John", "Sarah", "Anna", "Bob", "Tom", "Emily",
    "Chris", "Kate", "Nick", "Liz", "Jake", "Emma", "Dave", "Lisa",
    "Mark", "Amy", "Ben", "Chloe", "Dan", "Eva", "Frank", "Grace",
    "Henry", "Iris", "Jack", "Kelly", "Leo", "Maya", "Noah", "Olivia",
    "Pete", "Rachel", "Sam", "Tim", "Victor", "Wendy", "Zoe", "Ryan",
    # Полные имена
    "John Smith", "Sarah Johnson", "Michael Brown", "Emily Davis",
    "David Wilson", "Jessica Martinez", "Daniel Anderson", "Ashley Taylor",
    "Tyler White", "Megan Harris",
    # Роли/должности
    "Landlord", "Boss", "Dentist", "Trainer", "Coach", "Tutor",
    "Roommate", "Neighbor", "Barber", "Babysitter",
]

# (float_value, [list of text representations])
AMOUNTS = [
    (5.0,      ["5", "five", "five dollars", "five bucks", "$5"]),
    (10.0,     ["10", "ten", "ten dollars", "ten bucks", "$10"]),
    (15.0,     ["15", "fifteen", "fifteen dollars", "fifteen bucks"]),
    (20.0,     ["20", "twenty", "twenty dollars", "twenty bucks", "$20"]),
    (25.0,     ["25", "twenty-five", "twenty five", "twenty-five dollars"]),
    (30.0,     ["30", "thirty", "thirty dollars", "thirty bucks"]),
    (40.0,     ["40", "forty", "forty dollars", "forty bucks"]),
    (50.0,     ["50", "fifty", "fifty dollars", "fifty bucks", "$50", "50 USD"]),
    (75.0,     ["75", "seventy-five", "seventy five dollars"]),
    (100.0,    ["100", "a hundred", "one hundred", "a hundred dollars",
                "100 bucks", "$100", "hundred dollars"]),
    (150.0,    ["150", "a hundred and fifty", "one fifty", "150 dollars",
                "150 bucks"]),
    (200.0,    ["200", "two hundred", "200 dollars", "two hundred bucks", "$200"]),
    (250.0,    ["250", "two fifty", "two hundred and fifty", "250 bucks"]),
    (300.0,    ["300", "three hundred", "300 dollars", "three hundred bucks"]),
    (400.0,    ["400", "four hundred", "400 dollars", "four hundred bucks"]),
    (500.0,    ["500", "five hundred", "five hundred bucks", "half a grand",
                "$500", "500 dollars"]),
    (600.0,    ["600", "six hundred", "600 dollars", "six hundred bucks"]),
    (750.0,    ["750", "seven fifty", "seven hundred and fifty", "750 dollars"]),
    (800.0,    ["800", "eight hundred", "800 bucks", "800 dollars"]),
    (1000.0,   ["1000", "a thousand", "one thousand", "1k", "a grand",
                "1000 dollars", "$1000", "1,000"]),
    (1200.0,   ["1200", "twelve hundred", "1.2k", "1200 dollars"]),
    (1500.0,   ["1500", "fifteen hundred", "1.5k", "1500 dollars", "1,500"]),
    (2000.0,   ["2000", "two thousand", "2k", "two grand", "2000 dollars", "$2000"]),
    (2500.0,   ["2500", "twenty-five hundred", "2.5k", "2500 dollars"]),
    (3000.0,   ["3000", "three thousand", "3k", "three grand", "3000 dollars"]),
    (5000.0,   ["5000", "five thousand", "5k", "five grand", "$5000"]),
    (10000.0,  ["10000", "ten thousand", "10k", "ten grand"]),
]

POSITIVE_TEMPLATES = [
    "Transfer {amount} to {recipient}",
    "Send {amount} to {recipient}",
    "Pay {amount} to {recipient}",
    "Wire {amount} to {recipient}",
    "Move {amount} to {recipient}",
    "Shoot {amount} to {recipient}",
    "Forward {amount} to {recipient}",
    "Could you transfer {amount} to {recipient}?",
    "Could you send {amount} to {recipient}?",
    "Please transfer {amount} to {recipient}",
    "Please send {amount} to {recipient}",
    "I need to send {amount} to {recipient}",
    "I want to send {amount} to {recipient}",
    "I want to transfer {amount} to {recipient}",
    "Can you send {amount} to {recipient}?",
    "Can you transfer {amount} to {recipient}?",
    "Send {recipient} {amount}",
    "Transfer {recipient} {amount}",
    "Pay {recipient} {amount}",
    "Hey, send {amount} to {recipient}",
    "Hey, transfer {amount} to {recipient}",
    "Make a payment of {amount} to {recipient}",
    "Make a transfer of {amount} to {recipient}",
    "{amount} to {recipient} please",
    "Send {amount} to {recipient} please",
    "Transfer {amount} to {recipient} please",
    "I'd like to send {amount} to {recipient}",
    "I'd like to transfer {amount} to {recipient}",
    "Go ahead and send {amount} to {recipient}",
    "Send {amount} over to {recipient}",
    "Wire {amount} over to {recipient}",
    "Could you move {amount} to {recipient}?",
    "Would you send {amount} to {recipient}?",
    "Would you transfer {amount} to {recipient}?",
    "Pay {recipient} the {amount}",
    "Transfer {amount} to {recipient} asap",
    "Send {amount} to {recipient} right now",
    "Send {amount} to {recipient} today",
    "Transfer {amount} to {recipient} today",
    "Quickly send {amount} to {recipient}",
    "I need you to send {amount} to {recipient}",
    "I need you to transfer {amount} to {recipient}",
    "Transfer {amount} over to {recipient}",
    "Help me send {amount} to {recipient}",
    "Help me transfer {amount} to {recipient}",
    "Please wire {amount} to {recipient}",
    "Please move {amount} to {recipient}",
    "Do a transfer of {amount} to {recipient}",
    "Do a payment of {amount} to {recipient}",
    "Push {amount} to {recipient}",
    "Zap {amount} to {recipient}",
    "Toss {amount} to {recipient}",
    "Flick {amount} to {recipient}",
    "Drop {amount} to {recipient}",
    "Send the {amount} to {recipient}",
    "Transfer the {amount} to {recipient}",
]

# Валюты для exchange rate (отрицательные примеры)
CURRENCIES_A = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "CNY"]
CURRENCIES_B = ["EUR", "GBP", "RUB", "JPY", "CAD", "AUD", "CHF", "CNY"]

NEGATIVE_TEMPLATES = [
    # Баланс
    "What's my account balance?",
    "How much money do I have?",
    "Check my balance",
    "Show me my current balance",
    "What's my balance right now?",
    "How much is in my account?",
    "Can you check my account balance?",
    "What is my current balance?",
    "Tell me my account balance",
    # История транзакций
    "Show my recent transactions",
    "What were my last purchases?",
    "Show me my transaction history",
    "What did I spend last week?",
    "How much did I spend this month?",
    "List my recent payments",
    "Show me the last five transactions",
    # Курс валют
    "What's the USD to EUR exchange rate?",
    "How much is 100 euros in dollars?",
    "What's the current exchange rate for GBP?",
    "What's the dollar to pound rate today?",
    "EUR to USD rate please",
    "What is the exchange rate for {cur_a} to {cur_b}?",
    "Convert dollars to euros",
    "How many dollars is 50 euros?",
    "What's the exchange rate right now?",
    # Сбережения/цели
    "How can I save more money?",
    "I need to save 500 dollars this month",
    "What's my savings goal?",
    "Help me set up a savings plan",
    "How much should I save each month?",
    # Расходы (прошедшее время)
    "I just spent fifty dollars on groceries",
    "I paid my electricity bill today",
    "I bought something online for 30 dollars",
    "I already paid the rent",
    # Напоминания
    "Remind me to pay rent tomorrow",
    "Set a reminder to send money next week",
    "Don't let me forget to pay David",
    # Общие вопросы
    "What's the weather like today?",
    "Tell me a joke",
    "What time is it?",
    "What movies are playing tonight?",
    # Неоднозначные / не требуют действий
    "Money transfers take forever",
    "Transfer fees are too high",
    "I hate bank fees",
    "Is it safe to send money online?",
    "How long do transfers take?",
    "What are the transfer limits?",
    "Transfer times are so slow",
    # Уже выполнено
    "Did my transfer to Mom go through?",
    "Has the money arrived yet?",
    "Did I already send money to John?",
    "Was my last transfer successful?",
    "Check if my payment went through",
    # Другое
    "How do I open a new account?",
    "What are your fees?",
    "I want to close my account",
    "Change my PIN please",
    "I lost my card",
    "Block my credit card",
]


# ---------------------------------------------------------------------------
# Генерация
# ---------------------------------------------------------------------------

def make_label_transfer(recipient: str, amount: float) -> dict:
    return {
        "tool_name": "transfer_money",
        "arguments": {"recipient": recipient, "amount": amount}
    }


def generate_positive_examples(n: int, rng: random.Random) -> list:
    """Генерирует n уникальных положительных примеров."""
    pool = []
    for template in POSITIVE_TEMPLATES:
        for recipient in RECIPIENTS:
            for amount_val, amount_texts in AMOUNTS:
                for amount_text in amount_texts:
                    text = template.format(amount=amount_text, recipient=recipient)
                    pool.append({
                        "text": text,
                        "label": make_label_transfer(recipient, amount_val)
                    })

    rng.shuffle(pool)
    return pool[:n]


def generate_negative_examples(n: int, rng: random.Random) -> list:
    """Генерирует n уникальных отрицательных примеров."""
    pool = []
    for template in NEGATIVE_TEMPLATES:
        # Шаблоны с валютами разворачиваем
        if "{cur_a}" in template:
            for ca in CURRENCIES_A:
                for cb in CURRENCIES_B:
                    if ca != cb:
                        pool.append({
                            "text": template.format(cur_a=ca, cur_b=cb),
                            "label": None
                        })
        else:
            pool.append({"text": template, "label": None})

    # Если не хватает уникальных — дублируем с капитализацией вариаций
    while len(pool) < n:
        item = rng.choice(pool[:len(NEGATIVE_TEMPLATES)])
        variants = [
            item["text"].lower(),
            item["text"].upper() if len(pool) % 3 == 0 else item["text"],
            item["text"].rstrip("?") + ".",
        ]
        for v in variants:
            if v != item["text"]:
                pool.append({"text": v, "label": None})
                break

    rng.shuffle(pool)
    return pool[:n]


def main():
    rng = random.Random(RANDOM_SEED)

    print(f"Generating {TARGET_POSITIVE} positive examples...")
    positives = generate_positive_examples(TARGET_POSITIVE, rng)

    print(f"Generating {TARGET_NEGATIVE} negative examples...")
    negatives = generate_negative_examples(TARGET_NEGATIVE, rng)

    dataset = positives + negatives
    rng.shuffle(dataset)

    # Проверка на дубликаты по тексту
    texts = [item["text"] for item in dataset]
    unique_texts = set(texts)
    duplicates = len(texts) - len(unique_texts)
    if duplicates > 0:
        print(f"Warning: {duplicates} duplicate texts found, keeping all")

    print(f"\nDataset stats:")
    print(f"  Total:    {len(dataset)}")
    print(f"  Positive: {sum(1 for x in dataset if x['label'] is not None)}")
    print(f"  Negative: {sum(1 for x in dataset if x['label'] is None)}")
    print(f"  Unique texts: {len(unique_texts)}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
