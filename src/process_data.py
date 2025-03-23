import pandas as pd

def undersampling(input_path: str, output_path: str, label_column: str = 'label', sample_size: int = 1500):
    """라벨 0 데이터 수를 지정된 개수만큼 샘플링하여 클래스 불균형을 완화합니다."""

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
        print(f"Undersampled dataset saved to: {output_path}")

    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {input_path}")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    input_csv = "../data/train.csv"
    output_csv = "../data/preprocessed_train.csv"
    
    undersampling(input_csv, output_csv)
