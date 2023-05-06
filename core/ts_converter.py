import matplotlib.pyplot as plt
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
import face_recognition
from deepface import DeepFace
import numpy as np
import cv2
import os
import time
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf
from ffmpeg import input as ffin
import ffmpeg
import openai

# imports and constant preferences

openai.api_key = "KEY"
CHARACTERS_PATH = 'actors'
# manners of scene description
NEUTRAL = 'neutral'
SHORT = 'short'
HORROR = 'horror'
SCIENTIFIC = 'scientific'
EMOTIONAL = 'emotional'
KIDS = 'kids'
EXPERT = 'expert'


def extract_actors() -> list:
    """
    Function to extract face details for all of the characters in the given path (CHARACTERS_PATH constant used)
    :return: tuple of two lists: list of character name, list of face details, one sublist for each character
    """
    face_images = []
    face_character_names = []
    character_face_encodings = []
    for chc in os.listdir(CHARACTERS_PATH):
        curIm = cv2.imread(os.path.join(CHARACTERS_PATH, chc))
        face_images.append(curIm)
        face_character_names.append(os.path.splitext(chc)[0])
    for img in face_images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        character_face_encodings.append(encode)
    return face_character_names, character_face_encodings


def convert_time(milliseconds):
    seconds, milliseconds = divmod(int(milliseconds), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "%02d:%02d:%02d,%s" % (hours, minutes, seconds, str(milliseconds)[:3])


def deconvert_time(string_time):
    times = string_time.split(':')
    times.append(times[-1].split(',')[1])
    times[-2] = times[-2].split(',')[0]
    times = list(map(int, times))
    msec = times[-1] + 1000 * times[-2] + 60 * 1000 * times[-3] + 3600 * 1000 * times[-4]
    return msec


def create_string(start, text, length=1000 * 10):
    finish = start + length
    return f'{convert_time(start)} --> {convert_time(finish)}\n{text}'


def create_sub(input):
    return '\n\n'.join([str(idx + 1) + '\n' + create_string(x[0], x[1]) for idx, x in enumerate(input)])


def create_file_with_sub(data, name='subs.srt'):
    text_file = open(name, "w")
    n = text_file.write(create_sub(data))
    text_file.close()


def connect_scene_rec(option=0):
    if option == 0:
        model_name = "nlpconnect/vit-gpt2-image-captioning"
        tokenizer_model = "nlpconnect/vit-gpt2-image-captioning"
    else:
        model_name = "bipin/image-caption-generator"
        tokenizer_model = "gpt2"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, feature_extractor, tokenizer, device


def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=60,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()


def parse_audio_from_srt(srt_filename, processor, model, vocoder):
    with open(srt_filename) as f:
        strings = f.readlines()
    samplerate = 16000
    channels = 1
    with sf.SoundFile('subs.wav', 'w', samplerate, channels) as f:
        skip = False
        for st in strings:
            if '-->' in st:
                skip = False
                times = list(map(deconvert_time, st.split(' --> ')))
                current_time = f.tell() * 1000 / samplerate
                print(convert_time(f.tell()), convert_time(f.tell() / samplerate * 1000), convert_time(current_time))
                if current_time < times[0]:
                    duration = times[0] - current_time  # duration of the silent part in milliseconds
                    print(convert_time(duration), convert_time(times[0]))
                    samples = duration * samplerate // 1000  # number of samples in the silent part
                    zeros = np.zeros((int(samples), int(channels)))  # array of zeros
                    f.write(zeros)
                if current_time > times[0] + 20:
                    skip = True
            elif not skip and type(st) == str and not st.strip().isdigit() and len(st.strip()) > 2:
                inputs = processor(text=str(st), return_tensors="pt")
                print(st)

                # load xvector containing speaker's voice characteristics from a dataset
                embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
                speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

                speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
                f.write(speech.numpy())
    return 'subs.wav'


def write_to_ts(ts_filename, srt_filename, processor, model, vocoder):
    # Open the TS file
    input_file = ffmpeg.input(ts_filename)

    # Add an audio file
    audio_file = ffmpeg.input(parse_audio_from_srt(srt_filename, processor, model, vocoder))
    merged_audio = ffmpeg.filter([audio_file, input_file['a']], 'amix')

    # Combine the video and audio files
    output = ffmpeg.output(input_file['v'], merged_audio, 'output.ts')

    # Run the ffmpeg command to process the files
    ffmpeg.run(output)


def save_ts_subs(ts_filename, gap_sec=5, manner=NEUTRAL):
    model, feature_extractor, tokenizer, device = connect_scene_rec()
    char_names, char_encodings = extract_actors()
    capture = cv2.VideoCapture(ts_filename)
    if capture.isOpened() == False:
        print("Error opening the video file")
    else:
        fps = capture.get(cv2.CAP_PROP_FPS)
        print('Frames per second : ', fps, 'FPS')

        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        print('Frame count : ', frame_count)

        subs = []
        current_time = -1
        skip_frames = int(fps * gap_sec)  # skip frames for 3 seconds

        while (capture.isOpened()):
            temp_time = time.time()
            elapsed_time = temp_time - current_time
            if (elapsed_time >= 3 or current_time == -1):
                current_time = temp_time
                ret, frame = capture.read()
                if ret == False:
                    break
                moment = capture.get(cv2.CAP_PROP_POS_MSEC)
                max_length = 20
                num_beams = 4
                gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
                imgSmall = cv2.resize(frame, (0, 0), None, 0.75, 0.75)
                imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
                # skip frames
            FaceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2GRAY)
            face_locations = FaceCascade.detectMultiScale(gray, 1.1, 4)
            # face_locations = face_recognition.face_locations(imgSmall)
            current_frame_encodings = face_recognition.face_encodings(imgSmall, face_locations)

            pixel_values = feature_extractor(images=[imgSmall], return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

            output_ids = model.generate(pixel_values, **gen_kwargs)

            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]
            predtxt = [preds[0], [], '']
            for encoding, loc in zip(current_frame_encodings, face_locations):
                matches = face_recognition.compare_faces(char_encodings, encoding)
                distance_faces = face_recognition.face_distance(char_encodings, encoding)
                match_idx = np.argmin(distance_faces)
                analysis = DeepFace.analyze(imgSmall, enforce_detection=False, actions=("age", "gender", "emotion"))
                if analysis:
                    emotions = []
                    for el in analysis:
                        emotions.append(el['dominant_emotion'])
                    predtxt[2] = f'emotions of people on the frame: {", ".join(emotions)}'
                if matches[match_idx]:
                    name = char_names[match_idx]
                    gender = analysis[0]['gender']
                    age = analysis[0]['age']
                    predtxt[1].append(
                        f'mention character with the name {name} (of gender {gender}) of age {age} is on the screen')
            if manner == EXPERT:
                prompt = ("You are a film expert and you are talking to another film expert." +
                          " Describe a scene in the suitable terms, add details, be short." +
                          f"The general description of the frame: {predtxt[0]}")
            elif manner == KIDS:
                prompt = ("Describe the following scene for kids, exclude rude or bad words and violence." +
                          " Be short, concise and kind." +
                          f"The general description of the frame: {predtxt[0]}")
            elif manner == HORROR:
                prompt = ("You are creating subs for horror. Describe the following scene in a scary way." +
                          " You can paraphrase all the details ito make them more dark and violent. " +
                          f"The general description of the frame: {predtxt[0]}")
            elif manner == SCIENTIFIC:
                prompt = ("Describe the following scene in a scientific way with details" +
                          " Try to be short, but use complex words." +
                          f"The general description of the frame: {predtxt[0]}")
            elif manner == EMOTIONAL:
                prompt = ("Describe the following scene with the emphasis on the emotions of people in the frame" +
                          " Give emotional details, but be short." +
                          f"The general description of the frame: {predtxt[0]}")
            elif manner == SHORT:
                prompt = ("Describe the following scene shortly. " +
                          " Your description must consist less than 7 words. Focus on the main things, skip details." +
                          f"The general description of the frame: {predtxt[0]}")
            else:
                prompt = ("You are creating short audio description for a film." +
                          " I have passed you a frame. Describe the scene in a neutral, short and concise manner. " +
                          f"Skip emotions. The general description of the frame: {predtxt[0]}")
            if predtxt[1]:
                prompt += ', also'.join(predtxt[1])
            if predtxt[2]:
                prompt += predtxt[2]
            response = generate_response(prompt).split('\n')[-1]
            subs.append([moment, response])
            for i in range(skip_frames - 1):
                capture.grab()
            else:
                # skip frames
                for i in range(skip_frames - 1):
                    capture.grab()

    # Release the video capture object
    create_file_with_sub(subs)
    capture.release()
    cv2.destroyAllWindows()
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    modelsp = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    write_to_ts(ts_filename, '/content/subs.srt', processor, modelsp, vocoder)

