import asyncio
from interactions import Client, Intents, listen, slash_command, SlashContext, SlashCommand
import  interactions

from dotenv import load_dotenv
import os

from faster_whisper import WhisperModel

from HateSpeechDetector import HateBert

import io

model_size = "medium.en"

# Whisper model
model = WhisperModel(model_size, device="cuda", compute_type="float16")
hate_speech_detector = HateBert()

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
print(TOKEN)



bot = Client(intents=Intents.DEFAULT | Intents.MESSAGE_CONTENT | Intents.GUILD_MEMBERS | Intents.GUILD_PRESENCES | Intents.ALL, )

@bot.event()
async def on_ready():
    print(f'Logged in as {bot.user}')


def transcribe(audio):
  segments, info = model.transcribe(audio, beam_size=5, word_timestamps=True)
  transcribed_text = ""
  
  for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        transcribed_text+=segment.text

  return transcribed_text

async def real_time_trancription(voice_state: interactions.ActiveVoiceState, ctx: interactions.SlashContext):
    while voice_state.connected:
        await voice_state.start_recording()
        await asyncio.sleep(3)
        await voice_state.stop_recording()
        users = []
        messages = []
        for user_id, file in voice_state.recorder.output.items():
            audio_binary_io = io.BufferedReader(file)
            current_text = transcribe(audio_binary_io)
            if current_text == "": continue
            users.append(user_id)
            messages.append(current_text)
        responses = hate_speech_detector.prefict(messages)
        print(messages)
        print(responses)
        for i in range(len(responses)):
            if responses[i] == 1:
                member = ctx.guild.get_member(user_id)
                if member is not None:
                    await ctx.send(f"El usuario {member.mention} esta usando lenguaje ofensivo!")

    return


@slash_command(
        name="record",
        description="Bot records the audio for 5 seconds.",
)
async def record(ctx: interactions.SlashContext):
    await ctx.defer()

    voice_state = await ctx.author.voice.channel.connect()

    # Start recording
    await voice_state.start_recording()
    await asyncio.sleep(5)
    await voice_state.stop_recording()
    for user_id, file in voice_state.recorder.output.items():
        audio_binary_io = io.BufferedReader(file)
        transcribe(audio_binary_io)
        print(type(file))
    await ctx.send("Done")

@slash_command(
        name="join",
        description="Bot joins the voice channel the user is in."
)
async def join(ctx: interactions.SlashContext):
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        await channel.connect()
        await ctx.send(f"Joined {channel.name}")

        voice_state = await ctx.author.voice.channel.connect()
        asyncio.create_task(real_time_trancription(voice_state, ctx))

    else:
        await ctx.send("You are not in a voice channel!")



@slash_command(
        name="leave",
        description="Bot leaves the voice channel."
)
async def leave(ctx: interactions.SlashContext):
    if ctx.voice_state:
        await ctx.author.voice.channel.disconnect()
        await ctx.send("Left the voice channel.")
    else:
        await ctx.send("I'm not in a voice channel!")


@slash_command(
    name="hello",
    description="Bot greets the user."
)
async def hello(ctx: interactions.SlashContext):
    await ctx.send(f"Hello {ctx.author.mention}!")


@listen()  # this decorator tells snek that it needs to listen for the corresponding event, and run this coroutine
async def on_ready():
    # This event is called when the bot is ready to respond to commands
    print("Ready")
    print(f"This bot is owned by {bot.owner}")


@listen()
async def on_message_create(event):
    # This event is called when a message is sent in a channel the bot can see
    print(f"message received: {event.message.jump_url}")


bot.start(TOKEN)