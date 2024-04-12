import sys
sys.path.append("")
import random
import time
import streamlit as st
from src.utils import load_model, hf_model
from typing import Any


model_path = 'model_v1.pt'
model = load_model(model_path)
model_from_hf = hf_model()

dangerous_prompts = [
    'aim questions', 'all', 'all your', 'attention and', 'before speak', 'before that', 'behave not',
    'block', 'breakout', 'char', 'cleaning', 'context', 'cut off', 'developer', 'die', 'disregarding', 'do not',
    'documents', 'emojis', 'enabled', 'end product', 'euphony', 'everything', 'everything above', 'everything and',
    'everything before', 'everything write', 'extra', 'fabricated', 'fair', 'fancied', 'fictional', 'fictitious', 'foo',
    'forget', 'forget everything', 'frage', 'fuck', 'generate', 'happening in', 'hate', 'headline', 'hold on', 'if you',
    'igmre', 'igmre what', 'ignore', 'ignore all', 'imagine', 'imagine you', 'in your', 'incline', 'instruction',
    'instructions', 'instruktionen', 'instruktionen bitte', 'jailbreak', 'je', 'liu igmre', 'lo', 'login', 'mad',
    'make up', 'matter that', 'me that', 'now dan', 'of technology', 'on you', 'openai', 'order', 'pretend you',
    'print', 'prisonbreak', 'program line', 'prompt what', 'pwd', 'qui', 'remainder', 'report on', 'schreibe',
    'sentence the', 'spd', 'sql', 'state that', 'that what', 'this for', 'tokens will', 'translate', 'undecomposed',
    'vergiss', 'vergiss alles', 'verify and', 'vorherigen', 'vorherigen instruktionen', 'what you', 'who the',
    'you are', 'you don', 'you think', 'your instructions', 'zeit online', 'zon', 'bomb', 'virus', 'database'
]


def main() -> None:
    st.title("DOJOp")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Введите промпт"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    if prompt:
        with st.chat_message("assistant"):
            response1 = st.write_stream(response_generator(model, prompt))
            response2 = st.write_stream(hf_response_generator(model_from_hf, prompt))
        st.session_state.messages.append({"role": "assistant", "content": response1 + response2})


def response_generator(model_for_response: Any, prompt: str) -> None:
    if model_for_response.predict([prompt]):
        response = random.choice(
            [
                "Здесь используется какая-то уловка, будьте бдительны.",
                "Подозрительный текст, стоит его перепроверить.",
                "Возможна попытка jailbrek'а, проверьте.",
                "У этого промпта есть некоторые подозрительные признаки, будьте осторожны.",
                "Возможно, в этом запросе скрыта какая-то ловушка, будьте готовы к ней."
            ]
        )
    else:
        response = random.choice(
            [
                "Этот запрос выглядит стандартным и безопасным для использования.",
                "Нет признаков подозрительного поведения в этом запросе, он кажется безопасным.",
                "Все параметры этого запроса соответствуют стандартным безопасным показателям.",
                "На первый взгляд, этот запрос не вызывает подозрений и может быть использован без опасений.",
                "По всей видимости, этот запрос не представляет угрозы безопасности и может быть безопасно выполнен.",
                "Внешне этот запрос выглядит безопасным, нет необычных или подозрительных элементов.",
                "На основании предоставленной информации, этот запрос не вызывает "
                "сомнений относительно его безопасности.",
                "Вероятность возникновения проблем при выполнении этого запроса кажется низкой, его можно использовать.",
                "Все параметры и данные в этом запросе соответствуют стандартным безопасным критериям.",
                "Этот запрос был анализирован и не было выявлено никаких угроз безопасности, "
                "его можно считать безопасным.",
                "Промпт замечательный, используйте на здоровье.",
            ]
        )
    response = f'our model: {response}\n'
    if model_for_response.predict([prompt]):
        response = f'{response}\nПодозрительные слова в промпте:'
        for word in dangerous_prompts:
            if word in prompt:
                response += f'`{word}` '

    for word in response.split():
        time.sleep(0.1)
        yield word + ' '


def hf_response_generator(model_from_hf: Any, prompt: str) -> None:
    if model_from_hf.predict(prompt)[0]['label'] == 'INJECTION':
        response = random.choice(
            [
                "Здесь используется какая-то уловка, будьте бдительны.",
                "Подозрительный текст, стоит его перепроверить.",
                "Возможна попытка jailbrek'а, проверьте.",
                "У этого запроса есть некоторые подозрительные признаки, будьте осторожны.",
                "Возможно, в этом запросе скрыта какая-то ловушка, будьте готовы к ней."
            ]
        )
    else:
        response = random.choice(
            [
                "Этот запрос выглядит стандартным и безопасным для использования.",
                "Нет признаков подозрительного поведения в этом запросе, он кажется безопасным.",
                "Все параметры этого запроса соответствуют стандартным безопасным показателям.",
                "На первый взгляд, этот запрос не вызывает подозрений и может быть использован без опасений.",
                "По всей видимости, этот запрос не представляет угрозы безопасности и может быть безопасно выполнен.",
                "Внешне этот запрос выглядит безопасным, нет необычных или подозрительных элементов",
                "На основании предоставленной информации, этот запрос не вызывает "
                "сомнений относительно его безопасности.",
                "Вероятность возникновения проблем при выполнении этого запроса кажется низкой, его можно использовать.",
                "Все параметры и данные в этом запросе соответствуют стандартным безопасным критериям.",
                "Этот запрос был анализирован и не было выявлено никаких угроз безопасности, "
                "его можно считать безопасным.",
                "Промпт замечательный, используйте на здоровье.",
            ]
        )

    response = f'\nmodel from hugging face: \n{response}\n'
    for word in response.split():
        time.sleep(0.2)
        yield word + " "


def dist_bert():
    ...



if __name__ == '__main__':
    main()