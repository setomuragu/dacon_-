# dacon 대회
title: "데이콘 생육 환경 최적화 경진대회"

--------------------------------------------

csv 데이터와 이미지 데이터로 예측을 해야한다.<br><br>
하나의 데이터가 1행이 아닌 (1440, 21) 매트릭스로 주어져서 당황을 하였다.<br><br>
처음엔 하나의 행으로 만들기 위해 pca를 이용하여 1열로 만든 다음, 전치하여 1행으로 만들어 이미지 데이터를 열로 붙여 예측을 진행하였다.<br><br>
다음은 하나의 데이터를 (1, 1440 * 21)로 펼쳐서 예측을 진행하였다.<br><br>
결과는 예상과는 다르게 pca를 이용한 예측이 점수가 더 잘 나왔다.<br><br>
