import sys
sys.path.append("")
import random
import time
import streamlit as st
from src.utils import load_model
from typing import Any


model_path = 'model_v1.pt'
model = load_model(model_path)


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
            response = st.write_stream(response_generator(model, prompt))
        st.session_state.messages.append({"role": "assistant", "content": response})


def response_generator(model_for_response: Any, prompt: str) -> str:
    if model_for_response.predict([prompt]):
        response = random.choice(
            [
                "Здесь используется какая-то уловка, будьте бдительны",
                "Подозрительный текст, стоит его перепроверить",
                "Возможна попытка jailbrek'а, проверьте",
                "У этого запроса есть некоторые подозрительные признаки, будьте осторожны",
                "Возможно, в этом запросе скрыта какая-то ловушка, будьте готовы к ней"
            ]
        )
    else:
        response = random.choice(
            [
                "Этот запрос выглядит стандартным и безопасным для использования",
                "Нет признаков подозрительного поведения в этом запросе, он кажется безопасным",
                "Все параметры этого запроса соответствуют стандартным безопасным показателям",
                "На первый взгляд, этот запрос не вызывает подозрений и может быть использован без опасений",
                "По всей видимости, этот запрос не представляет угрозы безопасности и может быть безопасно выполнен",
                "Внешне этот запрос выглядит безопасным, нет необычных или подозрительных элементов",
                "На основании предоставленной информации, этот запрос не вызывает "
                "сомнений относительно его безопасности",
                "Вероятность возникновения проблем при выполнении этого запроса кажется низкой, его можно использовать",
                "Все параметры и данные в этом запросе соответствуют стандартным безопасным критериям",
                "Этот запрос был анализирован и не было выявлено никаких угроз безопасности, "
                "его можно считать безопасным",
                "Промт замечательный, используйте на здоровье",
            ]
        )
    for word in response.split():
        yield word + " "
        time.sleep(0.21)


if __name__ == '__main__':
    main()