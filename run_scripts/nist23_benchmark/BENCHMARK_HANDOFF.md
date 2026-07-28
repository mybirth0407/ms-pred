# NIST'23 스펙트럼 예측 벤치마크 — 에이전트 인수인계 문서

> 이 문서 하나로 다른 에이전트가 벤치마크 전체를 파악·재현·확장할 수 있도록 자립형으로 작성됨.
> 모든 경로는 저장소 루트(`/home/mybirth0407/workspaces/ms-pred`) 기준 상대경로.
> 작성 시점 커밋: `c680326` (main에 fast-forward 병합 완료).

---

## 0. TL;DR

- **무엇을**: ms-pred(Coley group MS/MS 스펙트럼 예측)의 학습 가능한 모델 4종을 NIST'23 데이터에 대해 **Bemis–Murcko scaffold split**(분자 누수 없음)으로 학습·평가.
- **평가 범위**: (1) 스펙트럼 예측 정확도, (2) 2-stage 모델의 stage-1 프래그먼트 회수율(P/R/F1), (3) retrieval(후보 랭킹) top-k.
- **핵심 결과 (테스트셋, cosine 유사도 기준 순위)**:

  | 순위 | 모델 | 유형 | Cosine | Entropy | Coverage | Retrieval top-1 / top-10 |
  |:---:|------|------|:---:|:---:|:---:|:---:|
  | 1 | **GLACIER** | joint (multi-GPU DDP) | **0.800** | 0.736 | 0.868 | **0.396 / 0.905** |
  | 2 | **ICEBERG 2.1** | 2-stage fragment DAG | **0.722** | 0.647 | 0.818 | 0.345 / 0.901 |
  | 3 | **MARASON** | 2-stage fragment + RAG¹ | **0.720** | 0.646 | 0.805 | 0.339 / 0.891 |
  | 4 | **MassFormer** | binned graph transformer | **0.512** | 0.486 | 0.710 | (미실행) |

  ¹ MARASON은 **base 모드(retrieval augmentation 없음)** 로 평가 — 자세한 내용은 §4, §8.

- **수행 기간**: **2026-07-22 ~ 07-23** (학습+전 평가; 모델별 상세 타임라인 §3.2).
- **데이터 전송**: 88G tarball(option B)이 원격 서버로 전송·크기검증 완료 (§7).
- **미완료**: SCARF, 3DMolMS, MassFormer retrieval, MARASON RAG-augmented (§8).

---

## 1. 환경 & 치명적 함정 (반드시 먼저 읽을 것)

### 1.1 파이썬/의존성
- **Python 3.12 전용**. base conda(3.13)를 쓰면 안 됨 — base가 다른 워크스페이스의 `ms_pred`를 shadow하고, 저장소가 `torch==2.6.0+cu124` / `torchdata<0.10`(cp313 wheel 없음)을 핀함.
- 환경 구성:
  ```bash
  cd /home/mybirth0407/workspaces/ms-pred
  uv sync --extra cu124 --extra test --python 3.12
  source .venv/bin/activate
  python -c "import torch, dgl, ms_pred; print(torch.__version__, dgl.__version__, ms_pred.__file__)"
  # torch 2.6.0+cu124, dgl 2.5.0+cu124, 8× RTX 6000 Ada (48GB)
  ```
- **학습/예측 전 항상 `source .venv/bin/activate`** — `run_from_config.py`가 `python3` 서브프로세스를 spawn하므로 venv가 PATH에 없으면 잘못된 인터프리터를 씀.

### 1.2 ⚠️ Editable install → MAIN/src (가장 중요한 함정)
- `.venv/lib/python3.12/site-packages/__editable__.ms_gen-0.1.0.pth` 안에 경로 하나(`/home/mybirth0407/workspaces/ms-pred/src`, 즉 **MAIN 체크아웃**)만 들어 있음.
- 그래서 `import ms_pred.*`는 **cwd/워크트리와 무관하게 항상 MAIN/src를 로드**함. 워크트리에서 `python3 src/...`를 실행해도 스크립트 파일 자체는 워크트리에서 읽지만, `import ms_pred`는 MAIN/src로 해석됨. (config는 상대경로라 워크트리에서 읽힘.)
- **규칙**:
  - 워크트리에서 수정한 `ms_pred` 코드를 런타임에 반영하려면 → **`PYTHONPATH=<worktree>/src`** 를 export하고 실행. (프래그먼트 모델 ICEBERG/MARASON/GLACIER의 각종 fix가 여기 의존.)
  - **MassFormer는 예외** — 컴파일된 Cython 확장 `algos2.so`가 MAIN/src에만 있으므로 **PYTHONPATH override 없이**(=MAIN/src 사용) 실행해야 함. worktree/src로 덮으면 `algos2` ImportError.
- 이 벤치마크의 모든 fix는 c680326으로 main에 병합됐으므로, **지금은 main 체크아웃에서 그냥 실행하면 fix가 반영됨**(PYTHONPATH 트릭 불필요). 위 내용은 워크트리 기반으로 재작업할 때를 위한 기록.

### 1.3 런타임 호환 fix (전부 c680326에 포함됨)
| 파일 | 수정 | 이유 |
|------|------|------|
| `src/ms_pred/common/misc_utils.py` (`safe_assign`) | `.astype(np.int64)` (was `np.integer`) | numpy≥2.0은 추상 dtype `np.integer` 금지. 프래그먼트 모델 예측 시 필수 |
| `src/ms_pred/glacier/joint_model.py:829` | `lr_scheduler_step(self, scheduler, optimizer_idx, metric=None)` | torch≥2.0 Lightning API 시그니처 불일치 |
| `src/ms_pred/marason/train_inten.py:621` | `strategy='ddp_find_unused_parameters_true'` | RAG reference param이 reference 비활성 시 미사용 → DDP 에러 |
| MassFormer chirality featurization | `safe_index` 사용 | rdkit 2025.03이 chirality tag 추가 |
| predict/hyperopt 스크립트 | `pl.seed_everything` 경로 | pytorch-lightning 2.6에서 이동 |

### 1.4 MassFormer 전용: algos2 컴파일
`uv sync`는 `algos2`를 빌드하지 않음. 한 번 빌드 필요(Python 3.12 dev header 필요):
```bash
cd src/ms_pred/massformer_pred/massformer_code
cython -3 algos2.pyx
gcc -O3 -fPIC -fopenmp -shared \
    -I<python3.12 include> -I"$(python -c 'import numpy;print(numpy.get_include())')" \
    algos2.c -o "algos2$(python -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')" -fopenmp
rm -f algos2.c && cd -
```

### 1.5 메모리(OOM) 주의
- GLACIER 4-GPU DDP에서 `num-workers`가 크면 host RAM OOM (num-workers × GPU수 = 실제 워커수). NIST'23 config는 `num-workers: [4]`로 고정(원래 32였음 → 128 워커로 OOM 발생하여 낮춤).
- 프로세스 kill 시 자기 자신을 패턴에 포함하지 말 것(self-pkill). `ps`로 PID 찾아 kill.

---

## 2. 데이터셋

### 2.1 규모 (scaffold_1.tsv 기준)
| Fold | 스펙트럼 | 분자(inchikey) |
|------|---:|---:|
| train | 141,477 | 32,648 |
| val | 17,688 | 5,264 |
| **test** | **17,686** | **9,291** |
| **합계** | **176,851** | **47,203** |

- 실제 평가된 테스트 스펙트럼은 ~17,565개(빈 스펙트럼/예측 실패분 제외).

### 2.2 Split
- **Bemis–Murcko scaffold split** (`data/spec_datasets/nist23/splits/scaffold_1.tsv`), seed 1, 컬럼 `Fold_0` (train/val/test 값).
- 분자 누수 없음 검증 완료: inchikey/connectivity가 2개 이상 fold에 걸치는 경우 0건.
- **random split(`split_1.tsv`)은 의도적으로 제외** — 무작위 분할은 train/test 간 구조 유사(analog/scaffold 공유)를 누수시켜 과대평가됨. scaffold split은 새로운 화학구조 일반화 능력(구조 규명에 중요한 지표)을 측정. `split_1.tsv` 파일은 저장소에 남아 있으므로 `iterative_args`/`test_entries` 엔트리를 다시 추가하면 재실행 가능.

### 2.3 데이터 파일 위치
```
data/spec_datasets/nist23/
├── labels.tsv                      # spec → 구조/adduct/instrument 메타 (컬럼: dataset,spec,name,formula,ionization,instrument,smiles,inchikey)
├── spec_files.hdf5                 # 실제 피크 데이터
├── mgf_files/                      # (원본 mgf)
├── splits/{split_1.tsv, scaffold_1.tsv}
├── subformulae/
│   ├── no_subform.hdf5             # 평가 gold (스펙트럼 정확도의 정답)
│   └── magma_subform_50/           # SCARF용 (아직 미완성일 수 있음)
├── magma_outputs/
│   ├── magma_tree.hdf5             # stage-1 fragment gold (MAGMa DAG)
│   └── ...
└── retrieval/
    ├── cands_df_scaffold_1_50.tsv       # retrieval 후보 리스트 (spec당 최대 50개: true + PubChem decoy)
    └── cands_pickled_scaffold_1_50.p
```
- ⚠️ `data/`는 `.gitignore`로 무시됨(git에 추적 안 됨). 파일은 디스크에 존재.
- retrieval 후보 원천: `data/retrieval/pubchem/pubchem_formula_map_nist23.p` (9.4GB, `/mnt/data1/metaboai/mspred/data/retrieval/pubchem/pubchem_formula_map.p`로의 심볼릭 링크).

---

## 3. 벤치마크한 모델 (4종)

모두 `scaffold_1_rnd1`(scaffold split, seed 1) 1회 학습. Early stopping patience=5 (ICEBERG/MARASON/GLACIER), MassFormer는 patience=20/max-epochs=20.

| 모델 | 구조 요약 | GPU | config |
|------|-----------|-----|--------|
| **ICEBERG 2.1** | 2-stage: 프래그먼트 DAG 생성(gen) → 강도 예측(inten) | 1 GPU | `configs/iceberg/nist23/` |
| **MARASON** | ICEBERG 계열 2-stage + RAG(reference) 확장; base 모드로 학습(`add-reference=false`) | 1 GPU | `configs/marason/nist23/` |
| **MassFormer** | binned graph transformer(그래프→고정 bin 스펙트럼) | 1 GPU | `configs/massformer/nist23/` |
| **GLACIER** | joint 모델(프래그먼트+강도 동시), multi-GPU DDP | 4 GPU (DDP) | `configs/glacier/nist23/joint_train_nist23.yaml` |

### 3.1 체크포인트 경로
```
ICEBERG   : results/iceberg_nist23/scaffold_1_rnd1/ckpt/gen/best.ckpt
            results/iceberg_nist23/scaffold_1_rnd1/ckpt/inten/best.ckpt
MARASON   : results/marason_nist23/scaffold_1_rnd1/ckpt/gen/best.ckpt
            results/marason_nist23/scaffold_1_rnd1/ckpt/inten/best.ckpt
GLACIER   : results/glacier_nist23/scaffold_1_rnd1/version_3/best.ckpt   ← version_3가 성공 런 (0~2는 폐기)
MassFormer: results/massformer_baseline_nist23/scaffold_1_rnd1/version_0/best.ckpt
```

### 3.2 수행 타임라인 (결과 파일 타임스탬프 기준)
전 작업이 **2026-07-22 ~ 07-23** 에 수행됨.

| 모델 | 학습 완료 | 스펙트럼 평가 | Stage-1 평가 | Retrieval |
|------|:---:|:---:|:---:|:---:|
| GLACIER | 2026-07-22 13:51 | 2026-07-22 18:19 | — | 2026-07-23 15:48 |
| ICEBERG 2.1 | 2026-07-22 20:32 | 2026-07-23 03:04 | 2026-07-23 06:57 | 2026-07-23 11:21 |
| MARASON | 2026-07-22 23:18 | 2026-07-23 06:01 | 2026-07-23 07:00 | 2026-07-23 22:18 |
| MassFormer | 2026-07-22 15:00 | 2026-07-22 17:19 | — | — |

- **"학습 완료"** = `best.ckpt` 저장 시각. **2-stage 모델(ICEBERG/MARASON)은 2단계(inten) 체크포인트 기준**이며, 1단계(gen)는 더 일찍 종료 — ICEBERG 07-22 05:46 / MARASON 07-22 05:47.
- **"평가"** = 결과 yaml 기록 시각. `best.ckpt`는 best epoch 저장 시점이라 실제 학습 종료보다 다소 이를 수 있음.
- GLACIER·MassFormer는 Stage-1/Retrieval 미실행(`—`).

---

## 4. 결과

### 4.1 스펙트럼 정확도 (테스트셋)
평가: `analysis/spec_pred_eval.py`, `--max-peaks 100 --min-inten 0`, gold=`subformulae/no_subform.hdf5`. SEM ≈ 0.0005–0.0009 (대규모 테스트셋).

| 모델 | Cosine | Entropy | Coverage |
|------|:---:|:---:|:---:|
| GLACIER | **0.8001** | 0.7363 | 0.8681 |
| ICEBERG 2.1 | 0.7221 | 0.6471 | 0.8178 |
| MARASON | 0.7202 | 0.6464 | 0.8054 |
| MassFormer | 0.5117 | 0.4862 | 0.7098 |

- **Cosine / Entropy** = 전체 스펙트럼 유사도(피크 위치 **및** 강도 모두 반영).
- **Coverage** = 예측이 m/z를 맞춘 참 피크 비율(위치만 보는 recall).
- 프래그먼트 기반(GLACIER/ICEBERG/MARASON)이 binned 모델(MassFormer)보다 확연히 우수.
- 개별 yaml: `results/<model>_nist23/scaffold_1_rnd1/preds/pred_eval.yaml` (+ 충돌에너지/이온/질량bin 그룹별 TSV).

### 4.2 Stage-1 프래그먼트 DAG 회수율 (2-stage 모델, 테스트셋)
stage-1 **생성기(generator)** 가 강도 예측 전, 참 MAGMa 프래그먼트 집합을 얼마나 회수하는지.
프래그먼트 동일성 = Weisfeiler–Leman graph hash (`--identity wl`, 원본 MAGMa `frag_hash`와 일치). 계산: `analysis/dag_frag_eval.py`.

| 모델 | Precision | Recall | F1 | Jaccard | 예측 프래그먼트 | 참 프래그먼트 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| ICEBERG 2.1 | 0.2650 | **0.7739** | 0.3569 | 0.2284 | 95.4 | 37.1 |
| MARASON | 0.2651 | **0.7750** | 0.3573 | 0.2287 | 95.4 | 37.1 |

- **Recall ≈ 0.77** — 미지 분자에서도 참 프래그먼트의 ~77%를 회수(완전성 양호).
- **Precision ≈ 0.27** — 생성된 ~95개 중 ~27%만 실제(평균 참 37개). threshold=0.0 / max-nodes=100에서 기대되는 값. max-nodes↓ / threshold↑ 하면 recall↔precision 트레이드오프.
- ICEBERG ≈ MARASON (동일 DAG 생성기 구조).
- 개별 yaml: `results/<model>_nist23/scaffold_1_rnd1/pred_eval_frag_test.yaml`.

### 4.3 Retrieval top-k (테스트셋, PubChem decoy, entropy 거리)
후보 리스트(`cands_df_scaffold_1_50.tsv`, spec당 true+decoy 최대 50개)에 대해 예측 스펙트럼 → 참값과의 거리로 랭킹. 세 모델 모두 동일한 ~17.5k 테스트 스펙트럼 평가.

| 모델 | top-1 | top-2 | top-3 | top-5 | top-10 | top-20 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **GLACIER** | **0.396** | 0.578 | 0.683 | 0.796 | **0.905** | 0.968 |
| ICEBERG 2.1 | 0.345 | 0.526 | 0.638 | 0.768 | 0.901 | 0.967 |
| MARASON | 0.339 | 0.519 | 0.634 | 0.761 | 0.891 | 0.964 |

- MassFormer retrieval은 미실행.
- ⚠️ **주의(비교 유효성)**: yaml의 `avg_total_decoys` 필드가 GLACIER=186.88 vs ICEBERG/MARASON=46.7로 달라 보이지만, 이는 **후보 수가 다른 게 아니라** GLACIER가 충돌에너지 변형별로 후보 예측을 카운트해서(≈46.7×4) 부풀려진 것. 실제 후보 분자는 세 모델 모두 동일(spec당 평균 46.7, 최대 50). 따라서 **top-k 랭킹 비교는 유효**함.
- 개별 yaml: `results/<model>_nist23/scaffold_1_rnd1/retrieval_nist23_scaffold_1_50/rerank_eval_entropy.yaml` (+ 그룹별 TSV). ⚠️ 이 yaml은 `individuals` 리스트가 방대함(GLACIER ~109MB) — 요약만 볼 땐 `grep -E '^avg_(top_[0-9]+|total_decoys|true_dist):'` 사용.

---

## 5. 파일 맵 (어디에 무엇이)

```
results/<model>_nist23/scaffold_1_rnd1/
├── ckpt/{gen,inten}/best.ckpt        # 2-stage 모델 체크포인트 (GLACIER는 version_3/best.ckpt, MassFormer는 version_0/best.ckpt)
├── args.yaml                         # 학습 하이퍼파라미터
├── *_train.log                       # 학습 로그
├── preds/
│   ├── pred_eval.yaml                # ★ 스펙트럼 정확도 (cosine/entropy/coverage)
│   └── *_grouped_*.tsv               # 충돌에너지/이온/질량bin 그룹별
├── pred_eval_frag_test.yaml          # ★ stage-1 프래그먼트 P/R/F1 (2-stage만)
├── preds_train_100/                  # 전체 데이터셋 예측 트리 (stage-1 평가 입력; 이름은 "train"이지만 test 포함 전체)
│   └── tree_preds.hdf5
└── retrieval_nist23_scaffold_1_50/
    ├── rerank_eval_entropy.yaml      # ★ retrieval top-k
    ├── preds*.hdf5 / binned_preds.hdf5   # 후보 예측 (재생성 가능, 대용량)
    └── *_grouped_*.tsv

run_scripts/nist23_benchmark/
├── README.md                # 전처리~실행 가이드 (환경/데이터 준비 상세)
├── RESULTS.md               # 결과 요약 (스펙트럼 + stage-1)
├── BENCHMARK_HANDOFF.md     # ← 이 문서
├── 00_preprocess.sh         # subformulae/MAGMa DAG/GLACIER 트리 생성 (멱등)
├── run_all_models.sh        # 6모델 순차 실행 + aggregate
└── stage1_frag_eval.sh      # stage-1 프래그먼트 평가 원샷

results/nist23_benchmark_summary.tsv   # 빠른 통합 표 (스펙트럼+stage-1)
analysis/dag_frag_eval.py              # stage-1 프래그먼트 P/R/F1 계산 (--split-name/--subset 필터 지원)
analysis/spec_pred_eval.py             # 스펙트럼 정확도 계산
src/ms_pred/retrieval/retrieval_benchmark.py   # retrieval top-k 계산
```

---

## 6. 재현 / 확장 방법

### 6.1 스펙트럼 정확도 표
```bash
python analysis/nist23_benchmark_aggregate.py   # → results/nist23_benchmark.tsv, .md
```

### 6.2 Stage-1 프래그먼트 P/R/F1 (ICEBERG/MARASON, test-only, CPU)
```bash
python analysis/dag_frag_eval.py \
  --pred-tree-h5 results/<model>_nist23/scaffold_1_rnd1/preds_train_100/tree_preds.hdf5 \
  --gold-tree-h5 data/spec_datasets/nist23/magma_outputs/magma_tree.hdf5 \
  --dataset nist23 --identity wl --split-name scaffold_1.tsv --subset test --max-cpu 16
```
(`preds_train_100/tree_preds.hdf5`는 이름과 달리 전체 데이터셋 예측을 담고 있어 `--subset test`로 테스트만 필터해야 함.)

### 6.3 Retrieval (예: ICEBERG)
```bash
# STEP1: 모든 후보에 대해 스펙트럼 예측
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/ms_pred/iceberg/predict_smis.py \
  --batch-size 64 --dataset-name nist23 --sparse-out --sparse-k 100 --max-nodes 100 \
  --split-name scaffold_1.tsv \
  --gen-checkpoint results/iceberg_nist23/scaffold_1_rnd1/ckpt/gen/best.ckpt \
  --inten-checkpoint results/iceberg_nist23/scaffold_1_rnd1/ckpt/inten/best.ckpt \
  --save-dir results/iceberg_nist23/scaffold_1_rnd1/retrieval_nist23_scaffold_1_50 \
  --dataset-labels data/spec_datasets/nist23/retrieval/cands_df_scaffold_1_50.tsv \
  --num-cpu-workers 64 --num-gpu-workers 16 --gpu --adduct-shift --binned-out
# STEP2: 랭킹 → top-k
python src/ms_pred/retrieval/retrieval_benchmark.py \
  --dataset nist23 --full-labels data/spec_datasets/nist23/retrieval/cands_df_scaffold_1_50.tsv \
  --formula-dir-name no_subform.hdf5 \
  --pred-file results/iceberg_nist23/scaffold_1_rnd1/retrieval_nist23_scaffold_1_50/preds.hdf5 \
  --num-bins 15000 --upper-limit 1500 --dist-fn entropy
```
- ⚠️ **MARASON은 `--binned-out`이 `binned_preds.hdf5`(≈44G)를 씀**(`preds.hdf5` 아님). STEP2에서 `--pred-file .../binned_preds.hdf5 --binned-pred` 지정.
- **GLACIER는 `predict_smis_joint.py`** 사용(체크포인트 `version_3/best.ckpt`, `--sparse-out --sparse-k 100`).
- 전체 원샷 스크립트는 job tmp에 있었음: `run_{iceberg,marason,glacier}_retrieval.sh` (§9 참고).

### 6.4 전체 파이프라인
```bash
bash run_scripts/nist23_benchmark/00_preprocess.sh          # 전처리(멱등)
bash run_scripts/nist23_benchmark/run_all_models.sh          # 전 모델 학습→예측→평가
# 또는 서브셋: bash run_scripts/nist23_benchmark/run_all_models.sh massformer molnetms
```

---

## 7. 다른 서버로 전송된 데이터

- **패키지**: option B(실용 88G tarball) — data + 4모델 체크포인트 + 전 평가 결과(yaml/tsv) + preds + retrieval top-k yaml + 후보 리스트. **재생성 가능한 retrieval 후보 예측 hdf5(~117G)는 제외**.
- **목적지**: `snu_won@59.150.33.1:/NHNHOME/WORKSPACE/26moe001_B/data/nist23_benchmark_B.tar`
  (접속: `ssh -i ~/.ssh/metabo-ai_key -p 39602 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null snu_won@59.150.33.1`)
- **검증**: 원본=목적지 `94,272,952,320 bytes` 정확히 일치. 전송 완료.
- **압축 해제**: `cd /NHNHOME/WORKSPACE/26moe001_B && tar xf data/nist23_benchmark_B.tar` → `data/ results/ run_scripts/` 생성.
- 제외된 retrieval 후보 예측 hdf5는 §6.3 STEP1으로 재생성 가능.

---

## 8. 미완료 / 다음 단계

| 항목 | 상태 | 필요 작업 |
|------|------|-----------|
| **SCARF** | 미실행 | `magma_subform_50` 필요. 알려진 upstream 버그: `scarf_pred/predict_gen.py`가 `form_preds/` 디렉토리에 JSON을 쓰는데 `data_scripts/forms/03_add_form_intens.py`는 `--pred-form-folder`를 HDF5 파일로 엶(ICEBERG 정확도 PR이 add_form_intens만 HDF5로 이전). SCARF stage2에서 실패하면 이 부분 fix 필요 |
| **3DMolMS** | 미실행 | 별도 런. aggregator에 슬롯 존재 |
| **MassFormer retrieval** | 미실행 | §6.3 방식으로 실행 가능(binned 모델이므로 `--binned-pred`) |
| **MARASON RAG-augmented** | 미실행 | 현재 결과는 **base 모드**(retrieval augmentation 없음). `add-reference=false`로 학습됐고 `--add-ref` 평가 경로는 precomputed nearest-neighbour store가 필요(아직 미구축). base MARASON ≈ ICEBERG(동일 생성기 계열). RAG 강화 수치는 reference store 구축 필요 |
| **contrastive finetuning** | 미적용 | 의도적 off — PubChem decoy로 retrieval 랭킹을 개선하지만 스펙트럼 정확도 벤치마크의 공정 비교를 위해 끔. ICEBERG/GLACIER train 스크립트 주석 참고 |
| **다중 seed / random split** | 미실행 | 현재 scaffold_1_rnd1 1회. 논문 그리드(3 seed × random+scaffold) 재현하려면 각 `configs/<model>/nist23/*.yaml`에 `iterative_args` + predict 드라이버에 `test_entries` 추가 |

---

## 9. 임시 드라이버 스크립트 (참고)

세션 중 사용한 원샷 오케스트레이션 스크립트들은 job tmp에 있었음(영구 저장소 아님):
`/home/mybirth0407/.claude/jobs/52d3fbbe/tmp/` — `run_{iceberg,marason,glacier,massformer}_*.sh`, `run_{...}_retrieval.sh`, `run_{...}_eval.sh`, `package_benchmark.sh` 등. 재현에는 §6의 명령 또는 `run_scripts/nist23_benchmark/*.sh`를 우선 사용.

---

## 10. 저장소 내 핵심 파일 (읽어볼 순서)

1. `run_scripts/nist23_benchmark/README.md` — 환경/데이터 준비(전처리, SDF/MSP 병합) 상세
2. `run_scripts/nist23_benchmark/RESULTS.md` — 결과 요약(스펙트럼+stage-1)
3. `docs/superpowers/specs/2026-07-21-nist23-benchmark-design.md` — 설계 문서
4. `analysis/dag_frag_eval.py`, `analysis/spec_pred_eval.py`, `src/ms_pred/retrieval/retrieval_benchmark.py` — 평가 진입점
5. `configs/{iceberg,marason,massformer,glacier}/nist23/*.yaml` — 학습 config

---

*커밋 `c680326` 기준. 코드 변경(호환 fix + stage-1 eval 도구)은 main에 병합됨. 원격 백업: `fork/worktree-nist23-benchmark` (github.com/mybirth0407/ms-pred).*
