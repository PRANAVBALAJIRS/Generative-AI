from dotenv import load_dotenv
import os
from aiogram import Bot, Dispatcher, executor, types
import openai
import sys

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

class Reference:
    def __init__(self) -> None:
        self.response = ""




reference = Reference()
model_name = "gpt-3.5-turbo"

#Initialize bot and dispatcher
bot = Bot(token = TELEGRAM_BOT_TOKEN)
dispatcher = Dispatcher(bot)

def clear_past():
    reference.response = ""

@dispatcher.message_handler(commands=['start'])
async def welcome(message: types.Message):
    await message.reply("Hi\nI am Tele Bot!\nCreated by Mlaverick. How can I assist you?")\
    
@dispatcher.message_handler(commands=['clear'])
async def clear(message: types.Message):
    clear_past()
    await message.reply("I've cleared the past conversation and context.")

@dispatcher.message_handler(commands=['help'])
async def helper(message: types.Message):
    help_command = """
    Hi there, I'm Telebot created by Mlaverick! Please follow these commands -
    /start - to start the conversation
    /clear - to clear the past conversation and context.
    /help - to get this help menu
    I hope this helps, :)
    """
    await message.reply(help_command)

@dispatcher.message_handler()
async def chatgpt(message: types.Message):
    print(f">>> USER: \n\t{message.text}")
    response = openai.ChatCompletion.create(
        model = model_name,
        messages = [
            {"role": "assistant", "content": reference.response},
            {"role": "user", "content": message.text}
        ],
        max_tokens = 25,
        temperature = 0.6
    )
    reference.response = response['choices'][0]['message']['content']
    print(f">>> chatGPT: \n\t{reference.response}")
    await bot.send_message(chat_id = message.chat.id, text = reference.response)

if __name__ == "__main__":
    executor.start_polling(dispatcher, skip_updates = True)