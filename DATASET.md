# DATASET.md

이 문서는 `data/train_data` 아래에 생성되는 학습 데이터의 구조를 설명합니다.

핵심 목표는 다음을 분명히 하는 것입니다.

- `frame`, `seq`, `version`이 각각 무엇인지
- `annotation`과 `binary_annotation`의 차이가 무엇인지
- 학습/검증/시각화 시 어떤 파일을 실제로 사용하는지
- `sam_feature`와 `class_embedding`이 무엇을 의미하는지

이 문서는 현재 코드 기준으로 작성되었습니다.
기준이 되는 주요 코드는 다음입니다.

- [ASI/tools/prepare_train_data.py](/mnt/work/doosik/ASI-Seg/ASI/tools/prepare_train_data.py)
- [ASI/dataset_audio.py](/mnt/work/doosik/ASI-Seg/ASI/dataset_audio.py)
- [ASI/train.py](/mnt/work/doosik/ASI-Seg/ASI/train.py)

## 1. 전체 개념

원본 데이터셋에서 한 장의 수술 영상 프레임을 가져오면, `train_data` 안에는 그 프레임에 대해 여러 종류의 파생 데이터가 저장됩니다.

- 원본 또는 augmentation된 RGB 이미지
- 클래스별 binary mask
- frame 전체에 대한 SAM feature
- 각 클래스 mask 영역에 대응하는 class embedding

즉 `train_data`는 단순 이미지 폴더가 아니라, 학습에 필요한 중간 산출물까지 포함한 "전처리 완료 데이터셋"입니다.

## 2. 용어 정리

### frame

`frame`은 영상의 한 장면, 즉 한 장의 이미지입니다.

예:

- `seq_5_frame238.png`
- `seq_9_frame299.png`

이 이름은 보통 다음 정보를 담습니다.

- 어떤 시퀀스에 속하는지
- 시퀀스 안에서 몇 번째 프레임인지

### seq

`seq`는 같은 수술 영상 또는 같은 연속 구간을 묶는 단위입니다.

예:

- `seq1`
- `seq5`
- `seq10`

폴더 구조에서 `seq`는 하위 디렉터리 이름으로 사용됩니다.

예:

```text
data/train_data/endovis_2017/0/images/seq5/seq_5_frame238.png
```

위 경로는:

- dataset: `endovis_2017`
- version: `0`
- sequence: `seq5`
- frame: `seq_5_frame238.png`

를 의미합니다.

### version

`version`은 augmentation 버전입니다.

- `version 0`: augmentation이 적용되지 않은 base 데이터
- `version 1 ~ N`: flip / scale / rotate / color jitter 등이 적용된 augmentation 버전

코드상 `prepare_train_data.py`는 `version 0`을 먼저 만들고, 그 다음 `version 1~40` 같은 추가 augmentation 버전을 생성합니다.

중요한 점:

- 같은 `frame`이라도 `version`이 다르면 이미지 내용이 달라질 수 있습니다.
- 예를 들어 `version 7`의 `seq_5_frame238.png`는 `version 0`의 같은 프레임과 좌우 반전되어 있을 수 있습니다.

## 3. 폴더 구조 개요

## EndoVis 2017

```text
data/train_data/endovis_2017/
  0/
    images/
    annotations/
    binary_annotations/
    sam_features_h/
    class_embeddings_h/
  1/
    images/
    binary_annotations/
    sam_features_h/
    class_embeddings_h/
  2/
    ...
```

## EndoVis 2018

```text
data/train_data/endovis_2018/
  train/
    0/
      images/
      annotations/
      binary_annotations/
      sam_features_h/
      class_embeddings_h/
      transcriptions/
    1/
      images/
      binary_annotations/
      sam_features_h/
      class_embeddings_h/
      transcriptions/
    2/
      ...
  val/
    images/
    annotations/
    binary_annotations/
    sam_features_h/
    class_embeddings_h/
    transcriptions/
```

차이를 보면 다음과 같습니다.

- `endovis_2017`은 최상위에 바로 `version` 폴더가 있습니다.
- `endovis_2018`은 먼저 `train/val`이 나뉘고, `train` 안에 다시 `version` 폴더가 있습니다.
- `val`은 augmentation하지 않으므로 `version` 폴더가 없습니다.

## 4. images, annotations, binary_annotations의 차이

### images

`images/seqX/frame.png`

이 파일은 실제 입력 RGB 프레임입니다.

- `version 0`에서는 원본 프레임
- `version > 0`에서는 augmentation된 프레임

예:

```text
data/train_data/endovis_2017/0/images/seq5/seq_5_frame238.png
data/train_data/endovis_2017/7/images/seq5/seq_5_frame238.png
```

두 파일은 같은 프레임 이름을 가지지만 내용은 다를 수 있습니다.

### annotations

`annotations/seqX/frame.png`

이 파일은 한 장의 multiclass GT mask입니다.

- 한 픽셀 값이 하나의 class id를 나타냅니다.
- 예를 들어 픽셀 값 `4`는 class 4 도구를 뜻합니다.

중요:

- 현재 코드에서는 base 데이터에 대해서만 `annotations`가 저장됩니다.
- augmentation version에는 `annotations`가 저장되지 않습니다.

즉 현재 구조상:

- `version 0`에는 multiclass annotation이 있음
- `version 1~40`에는 multiclass annotation이 없음

### binary_annotations

`binary_annotations/seqX/frame_classK.png`

이 파일은 한 프레임의 한 클래스만 분리한 binary GT mask입니다.

예:

```text
seq_5_frame238_class1.png
seq_5_frame238_class4.png
```

뜻:

- `seq_5_frame238_class1.png`: frame 238에서 class 1 도구 영역만 흰색(255), 나머지는 검정(0)
- `seq_5_frame238_class4.png`: 같은 frame에서 class 4 도구만 분리

즉 하나의 multiclass annotation은 여러 개의 binary annotation으로 분해됩니다.

관계는 다음과 같습니다.

```text
annotations/seq5/seq_5_frame238.png
    ->
binary_annotations/seq5/seq_5_frame238_class1.png
binary_annotations/seq5/seq_5_frame238_class4.png
binary_annotations/seq5/seq_5_frame238_class6.png
...
```

## 5. frame, annotation, binary_annotation의 관계

한 프레임을 기준으로 보면 다음과 같습니다.

### base version 예시

```text
images/seq5/seq_5_frame238.png
annotations/seq5/seq_5_frame238.png
binary_annotations/seq5/seq_5_frame238_class1.png
binary_annotations/seq5/seq_5_frame238_class4.png
binary_annotations/seq5/seq_5_frame238_class6.png
sam_features_h/seq5/seq_5_frame238.npy
class_embeddings_h/seq5/seq_5_frame238_class1.npy
class_embeddings_h/seq5/seq_5_frame238_class4.npy
class_embeddings_h/seq5/seq_5_frame238_class6.npy
```

의미:

1. `images/.../seq_5_frame238.png`
   실제 RGB frame
2. `annotations/.../seq_5_frame238.png`
   이 frame 전체에 대한 multiclass GT
3. `binary_annotations/..._classK.png`
   multiclass GT를 클래스별로 분해한 binary GT
4. `sam_features_h/.../seq_5_frame238.npy`
   이 frame 전체에서 추출한 SAM feature
5. `class_embeddings_h/..._classK.npy`
   class K 영역에 해당하는 feature 평균 벡터

## 6. version이 바뀌면 무엇이 같이 바뀌는가

augmentation version 생성 시 현재 코드에서는 다음이 같이 바뀝니다.

- `images`
- `binary_annotations`
- `sam_features_h`
- `class_embeddings_h`

즉 `version 7`의 frame과 `version 7`의 binary GT는 같은 augmentation을 공유합니다.

반면 현재 코드에서는 augmentation version에 대해 `annotations`는 저장하지 않습니다.

즉:

- 학습에는 문제 없음
- GUI에서 multiclass GT를 보려면 주의 필요

왜냐하면 학습은 `binary_annotations`를 직접 쓰기 때문입니다.

## 7. 학습 시 실제로 쓰는 GT는 무엇인가

학습에서 직접 쓰는 GT는 `annotations`가 아니라 `binary_annotations`입니다.

`Endovis17Dataset` / `Endovis18Dataset`의 train mode는 각 샘플마다 다음을 읽습니다.

- `sam_features_h/.../frame.npy`
- `binary_annotations/.../frame_classK.png`
- `class_embeddings_h/.../frame_classK.npy`

즉 학습 샘플의 단위는 "frame 전체"가 아니라:

- 한 frame
- 한 class

의 조합입니다.

예:

```text
seq_5_frame238_class4.png
```

은

- 입력 feature: `seq_5_frame238.npy`
- GT mask: `seq_5_frame238_class4.png`
- class embedding: `seq_5_frame238_class4.npy`

를 함께 사용하는 하나의 학습 샘플이 됩니다.

그래서 augmentation version에서도 학습이 정상 동작합니다.

이유:

- 입력 frame feature는 augmentation된 frame에서 추출됨
- GT binary mask도 같은 augmentation이 적용된 mask임

즉 학습 입력과 GT는 같은 좌표계에 있습니다.

## 8. validation / visualization에서 주의할 점

validation은 보통 base annotation 기준으로 평가하므로 `annotations`를 참조해도 괜찮습니다.

하지만 GUI 시각화에서 augmentation된 `images/version/...`를 보여주면서 GT로 `version 0`의 `annotations/...`를 사용하면 문제가 생깁니다.

예:

- 입력 이미지는 좌우 반전된 `version 7`
- GT는 반전되지 않은 `version 0`

이 경우 input과 GT overlay가 어긋나 보입니다.

즉 현재 구조에서는:

- `version 0`은 `images`와 `annotations`를 그대로 대응시킬 수 있음
- `version > 0`은 multiclass `annotations`가 없으므로 GUI에서 GT 비교 시 별도 처리가 필요함

## 9. sam_features_h란 무엇인가

`sam_features_h/seqX/frame.npy`

이 파일은 해당 frame 전체에 대해 SAM image encoder가 추출한 feature map입니다.

현재 코드 기준:

- `vit_h` 모델 사용
- SAM predictor로 이미지 전체를 넣어 feature 추출
- 저장 shape는 보통 `(64, 64, 256)`에 해당하는 형태

개념적으로는 다음과 같습니다.

- 원본 이미지를 바로 매번 SAM에 넣는 대신
- 한 번 feature를 미리 계산해 저장해 두고
- 학습/추론 때 재사용

장점:

- 학습 속도 향상
- image encoder 반복 계산 방지

## 10. class_embeddings_h란 무엇인가

`class_embeddings_h/seqX/frame_classK.npy`

이 파일은 class K의 mask 영역만 사용해 만든 대표 feature 벡터입니다.

생성 과정은 대략 다음과 같습니다.

1. frame 전체에서 SAM feature map 추출
2. class K의 binary mask를 SAM feature 해상도에 맞게 resize
3. mask가 1인 위치의 feature만 선택
4. 그 feature들을 평균내어 하나의 벡터 생성

즉 class embedding은

- "이 프레임에서 class K 도구가 차지하는 특징의 평균 표현"

입니다.

학습에서는 이 벡터가 contrastive loss의 reference embedding으로 사용됩니다.

## 11. 한 프레임이 여러 학습 샘플이 되는 이유

한 프레임 안에 클래스가 여러 개 있으면 그 수만큼 `binary_annotation`과 `class_embedding`이 생깁니다.

예를 들어 `seq_5_frame238.png` 안에 class 1, 4, 6이 있으면:

- `seq_5_frame238_class1.png`
- `seq_5_frame238_class4.png`
- `seq_5_frame238_class6.png`

이 생기고, 각 mask마다 대응하는 class embedding도 하나씩 생깁니다.

따라서 프레임 하나가 학습 샘플 여러 개로 확장됩니다.

## 12. EndoVis 2017과 2018의 차이

### EndoVis 2017

- `data/train_data/endovis_2017/<version>/...`
- `version 0`은 base
- `version 1~N`은 augmentation
- validation은 fold 기준으로 sequence를 나눔

### EndoVis 2018

- `data/train_data/endovis_2018/train/<version>/...`
- `data/train_data/endovis_2018/val/...`
- train만 augmentation version이 있음
- val은 augmentation 없이 고정
- transcription 파일도 함께 저장됨

## 13. 정리

가장 중요한 관계만 짧게 정리하면:

- `frame` = 한 장의 RGB 이미지
- `seq` = frame들이 속한 시퀀스 폴더
- `version` = augmentation 버전
- `annotations` = frame 전체에 대한 multiclass GT
- `binary_annotations` = frame의 multiclass GT를 클래스별로 쪼갠 binary GT
- `sam_features_h` = frame 전체에서 추출한 SAM feature map
- `class_embeddings_h` = 특정 class mask 영역의 feature 평균 벡터

그리고 현재 코드 기준으로 가장 중요한 주의점은 이것입니다.

- 학습은 `binary_annotations`를 사용하므로 augmentation version에서도 입력/GT 정렬이 맞음
- 하지만 GUI에서 `version > 0`의 input에 대해 `version 0`의 `annotations`를 겹치면 GT가 어긋나 보일 수 있음

## 14. 추천 후속 개선

시각화 도구에서 augmentation version까지 정확한 GT 비교를 하려면 두 가지 중 하나가 필요합니다.

1. augmentation version에도 `annotations/seqX/frame.png`를 저장한다.
2. GUI에서 `version > 0`일 때 GT overlay를 비활성화하거나 경고를 표시한다.

현재 구조에서는 1번이 가장 깔끔한 해결책입니다.
