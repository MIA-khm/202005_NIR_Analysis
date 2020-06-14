rlt_of_train
train data(train.csv)가 입력값인 출력 결과
decoder 결과가 아닌 encoder의 결과임

rlt_of_test
test data(test.csv)가 입력값인 출력 결과
decoder 결과가 아닌 encoder의 결과임

acc_df_src
autoencoder를 통해 encoding 후 decoding한 결과에 대한 정확도
정확도(mse)가 0에 가까울수록 입력(=출력)값으로 활용된 값으로 잘 복구됨

파일명의 숫자
추출되는 feature의 수. encoder의 최종단 node의 수