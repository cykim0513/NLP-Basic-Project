import argparse
import pandas as pd
import torch
import pytorch_lightning as pl
from train import Dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", nargs="+", default=["klue/roberta-large", "snunlp/KR-ELECTRA-discriminator"], type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epoch", default=1, type=int)
    parser.add_argument("--shuffle", default=True)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--train_path", default="../data/augmented_train.csv")
    parser.add_argument("--dev_path", default="../data/dev.csv")
    parser.add_argument("--test_path", default="../data/dev.csv")
    parser.add_argument("--predict_path", default="../data/test.csv")
    parser.add_argument("--num_workers", default=4, type=int)  # 데이터 로딩 최적화
    args = parser.parse_args()

    # 여러 모델과 Dataloader 생성
    dataloaders = []
    for model_name in args.model_names:
        dataloaders.append(Dataloader(model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                                      args.test_path, args.predict_path, args.num_workers))

    # 모델 경로와 모델 로드
    model_paths = ["model1.pt", "model2.pt"] 
    models = [torch.load(path) for path in model_paths]

    # Trainer 초기화
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.max_epoch, log_every_n_steps=1)

    predictions_list = []
    for model, dataloader in zip(models, dataloaders):
        dataloader.setup("test")  # 테스트 데이터셋 준비
        predictions = trainer.predict(model=model, datamodule=dataloader)
        predictions_list.append(torch.cat(predictions))

    # 앙상블(평균)
    predictions_mean = torch.mean(torch.stack(predictions_list), dim=0)

    # 최종 예측 저장
    final_predictions = list(round(float(i), 1) for i in predictions_mean)

    output = pd.read_csv("../data/sample_submission.csv")
    output["target"] = final_predictions
    output.to_csv("../data/output.csv", index=False)
    