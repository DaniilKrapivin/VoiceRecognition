import speech_recognition as sr
def tts(path):
   r = sr.Recognizer()
   srs = sr.AudioFile(path)
   with srs as source:
      audio = r.record(source)
   text = r.recognize_google(audio,language='ru-RU')
   return text
