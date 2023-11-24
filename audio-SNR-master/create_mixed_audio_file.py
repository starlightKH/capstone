# -*- coding: utf-8 -*-
import argparse #파일 파싱을 위한 모듈
import array
import math
import numpy as np 
import random  #array, math, numpy,random는 연산을 위한 모듈
import wave
import librosa, librosa.display 
import matplotlib.pyplot as plt   #librosa, librosa.display, matplotlib.pyplot는 시각화를 위한 모듈
from pydub import AudioSegment
import random
import subprocess


def get_args():
    parser = argparse.ArgumentParser()  #parser를 생성한다.
    parser.add_argument('--clean_file', type=str, required=True)    #clean_file, noise_file, output_mixed_file, snr을 입력받는다.
    parser.add_argument('--noise_file', type=str, required=True)
    parser.add_argument('--output_mixed_file', type=str, default='', required=True)
    parser.add_argument('--output_clean_file', type=str, default='')
    parser.add_argument('--output_noise_file', type=str, default='')    #output_clean_file, output_noise_file는 필수가 아니다.
    parser.add_argument('--snr', type=float, default='', required=True)     
    args = parser.parse_args()  #입력받은 인자를 args에 저장한다.
    return args            #args를 반환한다.

def cal_adjusted_rms(clean_rms, snr):   #clean_rms, snr을 입력받는다.
    a = float(snr) / 20     
    noise_rms = clean_rms / (10**a)     #noise_rms를 계산한다.
    return noise_rms

def cal_amp(wf):
    buffer = wf.readframes(wf.getnframes())     #buffer에 오디오 프레임을 저장한다.
    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)   
    return amptitude    
# wf.readframes(n)는 최대 n개의 오디오 프레임을 읽어들여 bytes 객체로 반환합니다. 
# wf.getnframes()는 오디오 프레임 수를 반환합니다. 
# 즉, wf.readframes(wf.getnframes())함수로 wav 파일의 모든 진폭값을 취득할 수 있습니다.
# 마지막으로 bytes 객체를 (np.frombuffer(buffer, dtype="int16")).astype(np.float64)함수를 사용해 np.float64에 캐스팅합니다.


def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))    #rms를 계산한다.

def save_waveform(output_path, params, amp):    #output_path, params, amp를 입력받는다.
    output_file = wave.Wave_write(output_path)  #output_file에 output_path를 저장한다.
    output_file.setparams(params)  #output_file의 파라미터를 설정한다.
    output_file.writeframes(array.array('h', amp.astype(np.int16)).tobytes() )     #output_file에 amp를 저장한다.
    output_file.close()

if __name__ == '__main__':
    FIG_SIZE = (15,10)  #FIG_SIZE를 (15,10)으로 설정한다.(그래프 크기)
    args = get_args()   

    clean_file = args.clean_file    #clean_file, noise_file, output_mixed_file, snr을 입력받는다.
    noise_file = args.noise_file

    clean_wav = wave.open(clean_file, "r")  #clean_wav, noise_wav에 clean_file, noise_file을 저장한다.
    noise_wav = wave.open(noise_file, "r")

    clean_amp = cal_amp(clean_wav)  #clean_amp, noise_amp를 계산한다.
    noise_amp = cal_amp(noise_wav)

    clean_rms = cal_rms(clean_amp)  #clean_rms를 계산한다.

    start = random.randint(0, len(noise_amp)-len(clean_amp))    #start에 0부터 noise_amp의 길이에서 clean_amp의 길이를 뺀 값 사이의 랜덤한 정수를 저장한다.
    divided_noise_amp = noise_amp[start: start + len(clean_amp)]    #divided_noise_amp에 start부터 start + len(clean_amp)까지의 값을 저장한다.
    noise_rms = cal_rms(divided_noise_amp)  #noise_rms를 계산한다.

    snr = args.snr  #snr을 저장한다.
    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)   #adjusted_noise_rms를 계산한다.
    
    adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / noise_rms)  #adjusted_noise_amp를 계산한다.
    mixed_amp = (clean_amp + adjusted_noise_amp)    #mixed_amp를 계산한다.

    #Avoid clipping noise
    max_int16 = np.iinfo(np.int16).max  #max_int16, min_int16을 계산한다.
    min_int16 = np.iinfo(np.int16).min
    if mixed_amp.max(axis=0) > max_int16 or mixed_amp.min(axis=0) < min_int16:     #두 진폭을 더한 값이 16비트를 넘어가면 절단현상이 일어나므로 이를 방지하기 위해 진폭을 조정한다.
        if mixed_amp.max(axis=0) >= abs(mixed_amp.min(axis=0)):     
            reduction_rate = max_int16 / mixed_amp.max(axis=0)
        else :
            reduction_rate = min_int16 / mixed_amp.min(axis=0)
        mixed_amp = mixed_amp * (reduction_rate)
        clean_amp = clean_amp * (reduction_rate)

    save_waveform(args.output_mixed_file, clean_wav.getparams(), mixed_amp)

    file = args.output_mixed_file
    # plt.figure(figsize=FIG_SIZE)
    # sig, sr = librosa.load(file, sr=16000)

    # time = np.arange(0, float(sig.shape[0]), 1) / sr

    # plt.plot(time, sig, label='Waveform', alpha=0.5)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Waveform')
    # plt.legend()
    # plt.show()


def mix_audio(original_audio_path, output_audio_path, total_duration=10000, segment_duration=2000):
    # 로드할 오디오 파일
    original_audio = AudioSegment.from_wav(original_audio_path)

    # 새로운 오디오 파일 생성
    new_audio = AudioSegment.silent(duration=total_duration)

    # 랜덤한 위치에 원본 오디오를 삽입
    start_time = random.randint(0, total_duration - segment_duration)
    new_audio = new_audio.overlay(original_audio, position=start_time)

    # 파일 저장
    new_audio.export(output_audio_path, format="wav")

# 사용 예시
# original_file_path = "C:\\Users\\rbgus\\Desktop\\capstone\\capstone\\test.wav"
# output_file_path = "C:\\Users\\rbgus\\Desktop\\capstone\\capstone\\output.wav"
# mix_audio(original_file_path, output_file_path)

    #실행 코드 (/capstone/audio-SNR-master에서 실행)
    # .\create_mixed_audio_file.py --clean_file C:\Users\USER\Desktop\capstone\capstone\audio-SNR-master\data\16_bit\source_clean\arctic_a0001.wav --noise_file C:\Users\USER\Desktop\capstone\capstone\audio-SNR-master\data\16_bit\source_noise\ch01.wav --output_mixed_file C:\Users\USER\Desktop\capstone\capstone\audio-SNR-master\data\16_bit\output_mixed\test.wav --snr 20

    #.\create_mixed_audio_file.py --clean_file D:\capstone file\clean.wav --noise_file D:\capstone file\noise.wav --output_mixed_file D:\capstone file\test.wav --snr 20
    #.\create_mixed_audio_file.py --clean_file C:\\Users\\rbgus\\Desktop\\capstone\\capstone\\clean.wav --noise_file C:\\Users\\rbgus\\Desktop\\capstone\\capstone\\noise.wav --output_mixed_file C:\\Users\\rbgus\\Desktop\\capstone\\capstone\\test.wav --snr 20
    #git bash에서 사용
    # python create_mixed_audio_file.py --clean_file C:\\Users\\rbgus\\Desktop\\capstone\\capstone\\clean.wav --noise_file C:\\Users\\rbgus\\Desktop\\capstone\\capstone\\noise.wav --output_mixed_file C:\\Users\\rbgus\\Desktop\\capstone\\capstone\\test.wav --snr 20
    #"D:\\capstone file\\교통소음\\130.도시 소리 데이터\\01.데이터\\2.Validation\\원천데이터\\VS_1.교통소음.zip\\1.자동차\\1.차량경적\\noise006.wav"
    #"D:\\capstone file\\교통소음\\130.도시 소리 데이터\\01.데이터\\2.Validation\\원천데이터\\VS_1.교통소음.zip\\1.자동차\\3.차량주행음\\bgs006.wav"
    #"D:\\capstone file\\output"
   # python create_mixed_audio_file.py --clean_file D:\\capstonefile\\noise\\noise006.wav --noise_file D:\\capstonefile\\bgs\\bgs004.wav --output_mixed_file D:\\capstone_file\\output\\test.wav --snr 20

command = "python create_mixed_audio_file.py --clean_file D:\\capstonefile\\noise\\noise006.wav --noise_file D:\\capstonefile\\bgs\\bgs004.wav --output_mixed_file D:\\capstone_file\\output\\test.wav --snr 20"
subprocess.run(command, shell=True)
# create_mixed_audio_file.py --clean_file D:\\capstonefile\\noise\\noise006.wav --noise_file D:\\capstonefile\\bgs\\bgs004.wav --output_mixed_file D:\\capstone_file\\output\\test.wav --snr 20