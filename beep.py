#!/usr/bin/python3 -u

from telethon import TelegramClient, events
import sys
import time

api_id = 25315069
api_hash = '419b7cd9f055a855ffd2f06948ab882e'

client = TelegramClient('beep', api_id, api_hash)

async def main():

    if len(sys.argv) < 2:
        print("Использование: python script.py <число1> <число2> ...")
        return
    
    numbers = [int(arg) for arg in sys.argv[1:]]
    
    # Getting entities through their ID (User, Chat or Channel)
#    entity = await client.get_entity(5569117599)
    # Does it have a username? Use it!
    entity = await client.get_entity('LinuxGodsWorkaholicBot')

    for i, count in enumerate(numbers):
        # Вызываем функцию count раз с интервалом 1 секунда
        for _ in range(count):
#            do_smthng()
#            await client.send_message('me', 'beep')
            await client.send_message(entity, str(count))
            time.sleep(5)
        
        # Ждем 3 секунды перед следующим аргументом (кроме последнего)
        if i < len(numbers) - 1:
            time.sleep(30)

    # Now you can use all client methods listed below, like for example...
#    await client.send_message('me', 'Hello to myself!')

with client:
    client.loop.run_until_complete(main())

