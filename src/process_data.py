import pandas as pd

def undersampling(input_path: str, output_path: str, label_column: str = 'label', sample_size: int = 1500):
    """
    Undersampling 함수
    - 특정 라벨 (기본값: 0) 의 샘플 수를 지정된 개수만큼 제한하여 균형을 맞춤
    - 파일을 불러오고, Undersampling을 수행한 후, 지정된 경로에 저장

    Args:
        input_path (str): 입력 CSV 파일 경로
        output_path (str): Undersampling된 데이터를 저장할 경로
        label_column (str): 라벨 컬럼명 (기본값: 'label')
        sample_size (int): 샘플링할 라벨 0 데이터 개수 (기본값: 1500)
    """

    try:
        # 데이터 로드
        df = pd.read_csv(input_path)

        if label_column not in df.columns:
            raise ValueError(f"'{label_column}' 컬럼이 데이터에 없습니다.")

        # 라벨 0 샘플링
        df_label_0_sampled = df[df[label_column] == 0][:sample_size].copy()

        # 라벨 0이 아닌 데이터 유지
        df_filtered = df[df[label_column] != 0].copy()

        # Undersampling 데이터 결합
        df_balanced = pd.concat([df_filtered, df_label_0_sampled])

        # 결과 저장
        df_balanced.to_csv(output_path, index=False)
        print(f'Undersampled dataset saved to: {output_path}')

    except FileNotFoundError:
        print(f'파일을 찾을 수 없습니다: {input_path}')
    except Exception as e:
        print(f'오류 발생: {e}')

if __name__ == '__main__':
    input_csv = '../data/train.csv'
    output_csv = '../data/preprocessed_train.csv'
    
    undersampling(input_csv, output_csv)
