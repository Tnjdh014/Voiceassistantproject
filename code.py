#installed openAI whisper
#installed PYtorch, selected to run through my GPU
#installed choco for windows


import whisper

model = whisper.load_model("base")
result = model.transcribe("AUDIO") #<----- make sure file is in directory by either selecting folder onto explorer or vscode directory
print(result["text"])
