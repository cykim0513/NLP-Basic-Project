import urllib.request
import json
import pandas as pd
import random

random.seed(0)

def back_translate_papago(text, _from, _to):
    """역번역(파파고 API 이용)"""
    url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"

    # API 키를 직접 코드에 입력
    client_id = "your_client_id"
    client_secret = "your_client_secret"

    data = f"source={_from}&target={_to}&text={urllib.parse.quote(text)}&honorific=true"

    request = urllib.request.Request(url)
    request.add_header("X-NCP-APIGW-API-KEY-ID", client_id)
    request.add_header("X-NCP-APIGW-API-KEY", client_secret)

    try:
        response = urllib.request.urlopen(request, data=data.encode("utf-8"))
        response_body = json.loads(response.read())

        return response_body["message"]["result"]["translatedText"]
        
    except Exception as e:
        print(f"API 호출 중 오류 발생: {e}")
        return text  # 오류 발생 시 원래 문장 반환
  

def using_back_translate(text):
    """한국어 → 일본어 → 한국어 역번역 수행"""
    translated_jp = back_translate_papago(text, "ko", "ja")
    return back_translate_papago(translated_jp, "ja", "ko")


def swap_sentence(df):
    """문장 순서 변경 (라벨이 0이 아닌 경우만)"""
    df_switched = df.copy()
    df_switched["sentence_1"], df_switched["sentence_2"] = df["sentence_2"], df["sentence_1"]
    return df_switched[df_switched["label"] != 0]


def copied_sentence(df, min_samples=1500):
    """동일 문장 쌍 추가 (라벨이 0인 샘플 중 특정 개수 이상인 경우)"""
    copied_df = df[df['label'] == 0][min_samples:].copy()
    copied_df["sentence_1"] = copied_df["sentence_2"]
    copied_df["label"] = 5.0
    return copied_df


def apply_back_translation(df, target_labels=None):
    """역번역을 이용한 데이터 증강 (특정 라벨 값 대상)"""
    if target_labels is None:
        target_labels = [0.5, 1.5, 2.5, 4.5, 4.8]

    extreme_values = df[df["label"].isin(target_labels)].copy()
    
    # `apply()` 내에서 직접 역번역 적용
    extreme_values["sentence_1"] = extreme_values["sentence_1"].apply(lambda x: using_back_translate(x))
    extreme_values["sentence_2"] = extreme_values["sentence_2"].apply(lambda x: using_back_translate(x))

    return extreme_values


if __name__ == "__main__":
    input_csv = "../data/preprocessed_train.csv"
    output_csv = "../data/augmented_train.csv"

    df = pd.read_csv(input_csv)

    df_swapped = swap_sentence(df)
    df_copied = copied_sentence(df)
    df_back_translated = apply_back_translation(df)

    final_df = pd.concat([df, df_swapped, df_copied, df_back_translated])
    final_df.to_csv(output_csv, index=False)

    print(f"데이터 증강 완료. 결과 저장: {output_csv}")
