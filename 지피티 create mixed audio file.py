# 필요한 라이브러리를 가져옵니다.
import argparse  # 명령행 인수 처리를 위한 모듈
import array  # 배열을 다루기 위한 모듈
import math  # 수학 연산을 위한 모듈
import numpy as np  # 배열 및 수학 연산을 위한 모듈
import random  # 난수 생성을 위한 모듈
import wave  # WAV 오디오 파일을 다루기 위한 모듈

# 명령행에서 입력 매개변수를 처리하는 함수
def get_args():
    parser = argparse.ArgumentParser()  # ArgumentParser 객체 생성
    parser.add_argument('--clean_file', type=str, required=True)  # 클린 오디오 파일 경로 입력 (필수)
    parser.add_argument('--noise_file', type=str, required=True)  # 노이즈 오디오 파일 경로 입력 (필수)
    parser.add_argument('--output_mixed_file', type=str, default='', required=True)  # 혼합된 오디오 파일 경로 입력 (기본값 지정, 필수)
    parser.add_argument('--output_clean_file', type=str, default='')  # 클린 오디오의 출력 파일 경로 입력 (기본값 지정)
    parser.add_argument('--output_noise_file', type=str, default='')  # 노이즈 오디오의 출력 파일 경로 입력 (기본값 지정)
    parser.add_argument('--snr', type=float, default='', required=True)  # 원하는 신호 대 잡음 비율(SNR) 입력 (필수)
    args = parser.parse_args()  # 입력 매개변수 구문 분석 및 결과를 저장
    return args  # 파싱된 입력 매개변수를 반환

# 클린 오디오의 RMS 값을 기반으로 원하는 SNR에 맞게 조정된 노이즈의 RMS 값을 계산하는 함수
def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20  # SNR을 dB에서 비선형 비율로 변환
    noise_rms = clean_rms / (10**a)  # 조정된 RMS 값을 계산
    return noise_rms  # 조정된 노이즈 RMS 값을 반환

# WAV 오디오 파일에서 샘플을 읽어와 배열로 반환하는 함수
def cal_amp(wf):
    buffer = wf.readframes(wf.getnframes())  # WAV 파일에서 모든 프레임을 읽어옴
    # 데이터 형식은 pulse-code modulation (PCM)에 따라 다르며, 일반적으로 int16(16비트)를 사용
    amplitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)  # 데이터를 16비트에서 부동 소수점으로 변환
    return amplitude  # 변환된 오디오 샘플을 반환

# 오디오 신호의 RMS(Root Mean Square) 값을 계산하는 함수
def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))  # 입력 오디오 신호의 RMS 값을 계산하고 반환

# 오디오를 WAV 파일로 저장하는 함수
def save_waveform(output_path, params, amp):
    output_file = wave.Wave_write(output_path)  # 출력 WAV 파일 생성
    output_file.setparams(params)  # 출력 파일의 파라미터 설정 (nchannels, sampwidth, framerate, nframes, comptype, compname)
    output_file.writeframes(array.array('h', amp.astype(np.int16)).tobytes())  # 오디오 데이터를 파일에 쓰기
    output_file.close()  # 파일 닫기

if __name__ == '__main__':
    args = get_args()  # 명령행 입력 매개변수 파싱

    clean_file = args.clean_file  # 클린 오디오 파일 경로
    noise_file = args.noise_file  # 노이즈 오디오 파일 경로

    clean_wav = wave.open(clean_file, "r")  # 클린 오디오 WAV 파일 열기 (읽기 모드)
    noise_wav = wave.open(noise_file, "r")  # 노이즈 오디오 WAV 파일 열기 (읽기 모드)

    clean_amp = cal_amp(clean_wav)  # 클린 오디오 샘플을 읽어와서 배열로 변환
    noise_amp = cal_amp(noise_wav)  # 노이즈 오디오 샘플을 읽어와서 배열로 변환

    clean_rms = cal_rms(clean_amp)  # 클린 오디오의 RMS 값을 계산

    start = random.randint(0, len(noise_amp) - len(clean_amp))  # 시작 지점을 랜덤하게 선택하여 오디오를 자르기 위한 시작 위치 결정
    divided_noise_amp = noise_amp[start: start + len(clean_amp)]  # 노이즈 오디오를 클린 오디오와 같은 길이로 자름
    noise_rms = cal_rms(divided_noise_amp)  # 자른 노이즈 오디오의 RMS 값을 계산

    snr = args.snr  # 사용자가 지정한 신호 대 잡음 비율(SNR)
    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)  # 조정된 RMS 값 계산

    adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / noise_rms)  # 조정된 RMS에 따라 노이즈 오디오 조정
    mixed_amp = (clean_amp + adjusted_noise_amp)  # 클린 오디오와 조정된 노이즈 오디오를 혼합

    # 클리핑(음성 신호가 최대/최소값을 벗어나는 것) 방지
    max_int16 = np.iinfo(np.int16).max  # 16비트 정수의 최대값
    min_int16 = np.iinfo(np.int16).min  # 16비트 정수의 최소값
    if mixed_amp.max(axis=0) > max_int16 or mixed_amp.min(axis=0) < min_int16:
        if mixed_amp.max(axis=0) >= abs(mixed_amp.min(axis=0)): 
            reduction_rate = max_int16 / mixed_amp.max(axis=0)  # 최대값을 제한
        else :
            reduction_rate = min_int16 / mixed_amp.min(axis=0)  # 최소값을 제한
        mixed_amp = mixed_amp * (reduction_rate)  # 혼합된 오디오를 감소된 비율로 조정
        clean_amp = clean_amp * (reduction_rate)  # 클린 오디오도 동일한 비율로 조정

    save_waveform(args.output_mixed_file, clean_wav.getparams(), mixed_amp)  # 혼합된 오디오를 WAV 파일로 저장
