# -*- coding: utf-8 -*-
import argparse
import numpy as np
import random
import soundfile as sf
from enum import Enum

class EncodingType(Enum):
    def __new__(cls, *args, **kwds):
        # 각 멤버의 값(value)를 설정합니다. 새로운 멤버가 추가될 때마다 4씩 증가합니다.
        value = len(cls.__members__) + 4
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, dtype, description, subtype, maximum, minimum):
        # 각 멤버의 속성을 설정합니다.
        self.dtype = dtype  # 데이터 유형 (예: "int16", "float32")
        self.description = description  # 인코딩 유형에 대한 설명
        self.subtype = subtype  # 서브타입 (예: "PCM_16", "FLOAT")
        self.maximum = maximum  # 데이터 유형의 최대값
        self.minimum = minimum  # 데이터 유형의 최소값
    
    # 다양한 인코딩 유형을 정의합니다. 각각의 유형은 튜플로 표현되며, 다음과 같은 정보를 포함합니다.
    INT16 = (
        "int16",  # 데이터 유형
        "Signed 16 bit PCM",  # 인코딩 유형에 대한 설명
        "PCM_16",  # 서브타입
        np.iinfo(np.int16).max,  # 16비트 정수의 최대값
        np.iinfo(np.int16).min,  # 16비트 정수의 최소값
    )
    INT32 = (
        "int32",
        "Signed 32 bit PCM",
        "PCM_32",
        np.iinfo(np.int32).max,  # 32비트 정수의 최대값
        np.iinfo(np.int32).min,  # 32비트 정수의 최소값
    )
    FLOAT32 = (
        "float32",
        "32 bit float",
        "FLOAT",
        1,  # 부동 소수점의 최대값
        -1,  # 부동 소수점의 최소값
    )
    FLOAT64 = (
        "float64",
        "64 bit float",
        "DOUBLE",
        1,  # 부동 소수점의 최대값
        -1,  # 부동 소수점의 최소값
    )




import argparse  # 명령행 인수 처리를 위한 모듈
import numpy as np  # 배열 및 수학 연산을 위한 모듈
import soundfile as sf  # 오디오 파일 입출력을 위한 모듈

# 명령행에서 입력 매개변수를 처리하는 함수
def get_args():
    parser = argparse.ArgumentParser()  # ArgumentParser 객체 생성
    parser.add_argument("--clean_file", type=str, required=True)  # 클린 오디오 파일 경로 입력 (필수)
    parser.add_argument("--noise_file", type=str, required=True)  # 노이즈 오디오 파일 경로 입력 (필수)
    parser.add_argument("--output_mixed_file", type=str, default="", required=True)  # 혼합된 오디오 파일 경로 입력 (기본값 지정, 필수)
    parser.add_argument("--output_clean_file", type=str, default="")  # 클린 오디오의 출력 파일 경로 입력 (기본값 지정)
    parser.add_argument("--output_noise_file", type=str, default="")  # 노이즈 오디오의 출력 파일 경로 입력 (기본값 지정)
    parser.add_argument("--snr", type=float, default="", required=True)  # 원하는 신호 대 잡음 비율(SNR) 입력 (필수)
    args = parser.parse_args()  # 입력 매개변수 구문 분석 및 결과를 저장
    return args  # 파싱된 입력 매개변수를 반환

# 클린 오디오의 RMS 값을 기반으로 원하는 SNR에 맞게 조정된 노이즈의 RMS 값을 계산하는 함수
def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20  # SNR을 dB에서 비선형 비율로 변환
    noise_rms = clean_rms / (10 ** a)  # 조정된 RMS 값을 계산
    return noise_rms  # 조정된 노이즈 RMS 값을 반환

# 오디오 신호의 RMS(Root Mean Square) 값을 계산하는 함수
def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))  # 입력 오디오 신호의 RMS 값을 계산하고 반환

# 오디오를 지정된 형식과 서브타입으로 파일에 저장하는 함수
def save_waveform(output_path, amp, samplerate, subtype):
    sf.write(output_path, amp, samplerate, format="wav", subtype=subtype)  # 주어진 경로에 오디오 저장





if __name__ == "__main__":
    # 명령행에서 입력 매개변수를 가져옵니다.
    args = get_args()

    # 클린 오디오 파일과 노이즈 오디오 파일 경로를 변수에 저장합니다.
    clean_file = args.clean_file
    noise_file = args.noise_file

    # 클린 오디오 파일의 메타데이터를 가져옵니다.
    metadata = sf.info(clean_file)

    # 클린 오디오의 인코딩 유형을 결정합니다.
    for item in EncodingType:
        if item.description == metadata.subtype_info:
            encoding_type = item

    # 클린 오디오와 노이즈 오디오를 읽어옵니다.
    clean_amp, clean_samplerate = sf.read(clean_file, dtype=encoding_type.dtype)
    noise_amp, noise_samplerate = sf.read(noise_file, dtype=encoding_type.dtype)

    # 클린 오디오의 RMS 값을 계산합니다.
    clean_rms = cal_rms(clean_amp)

    # 노이즈 오디오 중에서 임의의 위치에서 클린 오디오와 같은 길이의 노이즈를 추출합니다.
    start = random.randint(0, len(noise_amp) - len(clean_amp))
    divided_noise_amp = noise_amp[start : start + len(clean_amp)]
    noise_rms = cal_rms(divided_noise_amp)

    # 입력으로 받은 SNR을 이용하여 조정된 노이즈의 RMS 값을 계산합니다.
    snr = args.snr
    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)

    # 조정된 노이즈를 계산하고 클린 오디오에 더하여 혼합된 오디오를 생성합니다.
    adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / noise_rms)
    mixed_amp = clean_amp + adjusted_noise_amp

    # 클리핑을 방지하기 위해 혼합된 오디오가 최대/최소 임계값을 벗어나면 조절합니다.
    max_limit = encoding_type.maximum
    min_limit = encoding_type.minimum
    if mixed_amp.max(axis=0) > max_limit or mixed_amp.min(axis=0) < min_limit:
        if mixed_amp.max(axis=0) >= abs(mixed_amp.min(axis=0)):
            reduction_rate = max_limit / mixed_amp.max(axis=0)
        else:
            reduction_rate = min_limit / mixed_amp.min(axis=0)
        mixed_amp = mixed_amp * (reduction_rate)
        clean_amp = clean_amp * (reduction_rate)

    # 혼합된 오디오를 지정된 파일에 저장합니다.
    save_waveform(
        args.output_mixed_file, mixed_amp, clean_samplerate, encoding_type.subtype
    )
