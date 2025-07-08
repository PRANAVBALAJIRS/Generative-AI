import chainlit as cl
from src.llm import ask_order, messages

@cl.on_message
async def main(message: cl.Message):
    messages.append({"role": "user", "content": message.content})
    response = ask_order(messages)
    messages.append({"role": "assistant", "content": response})


    await cl.Message(
        content = response,
    ).send()