# New RecSim

Google RecSim을 기반으로 한 추천 시뮬레이터입니다. NumPy 기반의 세 환경과 기본 에이전트, Gymnasium 인터페이스, 선택 설치하는 PyTorch FullSlateQ·SlateDecompQ 학습 경로를 제공합니다.

Python **3.12–3.14**, NumPy 2, Gymnasium 1.x를 대상으로 검증합니다. 설치 배포명은 `new-recsim`, Python import 이름은 `recsim`입니다. 이 workspace 버전은 아래처럼 로컬 소스에서 설치합니다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

기본 설치는 시뮬레이션·CSV/JSONL 출력·기본 에이전트·runner를 포함합니다. PyTorch 학습과 TensorBoard가 필요하면 다음을 추가합니다.

```bash
# CPU에서 사용할 경우 먼저 CPU wheel을 설치합니다.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[rl]'
```

개발 도구는 `.[dev]`, 노트북 실행 도구는 `.[docs]`로 설치합니다. 실제 검증 버전은 [검증 기록](docs/refactoring/validation.md)과 [constraints](constraints/README.md)에 있습니다.

## 환경 사용

```python
from recsim.environments import interest_evolution

env = interest_evolution.create_environment(
    dict(num_candidates=5, slate_size=2, seed=7, resample_documents=True)
)
try:
    observation, info = env.reset(seed=7)
    observation, reward, terminated, truncated, info = env.step([0, 2])
    print(reward, info["action_document_ids"])
finally:
    env.close()
```

`interest_evolution`, `interest_exploration`, `long_term_satisfaction`이 같은 reset/step 계약을 사용합니다. `observation['doc']` 키는 항상 `"0"`부터 `"N-1"`까지의 후보 슬롯입니다. 행동은 슬롯 인덱스이며, 실제 ID는 `info['document_ids']`, 실행한 행동의 ID는 `info['action_document_ids']`에 있습니다. 초기 response도 선언된 공간에 포함됩니다.

`reset(seed=...)`는 환경 내부의 사용자·문서·선택·상태 전이 난수를 재설정합니다. seed 없는 후속 reset은 기존 난수 흐름을 이어갑니다. 다중 사용자 wrapper의 reward는 합계이고, 사용자별 값과 종료 상태는 `info`에 있습니다.

## 시뮬레이션 데이터 출력

```python
from recsim.simulation import simulate_users_csv, simulate_users_json

config = dict(
    slate_size=2,
    num_candidates=5,
    num_users=3,
    steps=10,
    global_seed=42,
    sim_seed=1,
    progress=False,
)
simulate_users_csv(file_name="outputs/simulation.csv", **config)
simulate_users_json(file_name="outputs/simulation.jsonl", **config)
```

기존 `from simulate_api import simulate_users_csv, simulate_users_json`도 지원합니다. CSV는 한 행에 한 step, JSONL은 한 줄에 한 사용자의 `steps` 배열을 저장합니다. 기본 `steps`는 CSV **100**, JSONL **20**이므로 비교할 때 명시적으로 맞추세요.

| 필드 | 의미 |
| --- | --- |
| `user_id`, `step` | 생성한 사용자 번호, 0부터 시작하는 step |
| `user_*` / `user` | 해당 행동을 수행한 **이후**의 사용자 관측 |
| `action` | 쉼표로 연결한 후보 슬롯 인덱스 문자열 |
| `reward` | 환경 reward; 기본 interest evolution은 clicked watch time |
| `resp_i_*` / `response` | click, click_doc_id, watch_time, liked, quality, cluster_id |

CSV header는 사용자 feature 수와 slate 크기에서 생성합니다. `<출력파일>.metadata.json`에 seed·환경 설정·schema·패키지 버전을 기록합니다(`metadata=False`로 생략 가능). 기존 export는 interest evolution 전용이며, 다른 환경이나 행동 전후 상태·종료 flag가 필요하면 `recsim.simulation.iter_episode()`를 사용하세요. interest evolution에서 재샘플링하는 동일 ID의 feature는 달라질 수 있습니다.

## 학습·재개·평가

기본 설치의 random agent로 runner를 실행할 수 있습니다.

```bash
python -m recsim.main --agent_name random --base_dir outputs/random
```

PyTorch 설치 후 FullSlateQ 예제:

```bash
python -m recsim.main --agent_name full_slate_q \
  --base_dir outputs/full_slate_q --gin_files examples/small.gin \
  --num_candidates 5 --slate_size 2 --num_iterations 2 \
  --max_training_steps 40 --max_steps_per_episode 20 \
  --episode_log_file steps.jsonl
```

기본 `--mode both`는 학습 후 평가합니다. 같은 설정·경로에 `--num_iterations 3`을 전달하면 저장된 iteration 이후를 실행합니다. `--mode eval`은 최신 체크포인트만 평가하며, 체크포인트가 없으면 오류를 반환합니다. `max_training_steps`는 iteration당 최소 step 수이며 episode 경계까지 실행하므로 초과할 수 있습니다.

`--agent_name slate_decomp_q`는 SlateDecompQ를 선택합니다. FullSlateQ는 순서 있는 slate를 열거하므로 후보 수를 작게 유지하세요. `--tensorboard`는 `rl` extra가 필요합니다. 로그는 UTF-8 JSONL, 체크포인트는 `train/checkpoints/ckpt_<iteration>.pkl`입니다. 이 체크포인트는 학습 상태와 RNG를 포함한 pickle이므로 신뢰하는 파일만 로드하세요. 기존 TensorFlow 체크포인트 자동 변환은 지원하지 않습니다.

## 개발·문서

- [기여 및 검증 명령](CONTRIBUTING.md)
- [마이그레이션: API·RNG·출력·체크포인트](docs/migration.md)
- [API 문서](docs/api_docs/python/index.md)
- [리팩터링 계획 및 진행 상태](plan.md), [로컬 검증 기록](docs/refactoring/validation.md)
- 노트북: [개요](recsim/colab/RecSim_Overview.ipynb), [환경 개발](recsim/colab/RecSim_Developing_an_Environment.ipynb), [에이전트 개발](recsim/colab/RecSim_Developing_an_Agent.ipynb)

## 원 프로젝트 및 인용

RecSim의 설계는 [Ie et al., RecSim: A Configurable Simulation Platform for Recommender Systems](https://arxiv.org/abs/1909.04847)에 설명되어 있습니다. 연구에 사용하면 원 논문을 인용해 주세요.

```bibtex
@article{ie2019recsim,
    title={RecSim: A Configurable Simulation Platform for Recommender Systems},
    author={Eugene Ie and Chih-wei Hsu and Martin Mladenov and Vihan Jain and Sanmit Narvekar and Jing Wang and Rui Wu and Craig Boutilier},
    year={2019},
    eprint={1909.04847},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

[Apache License 2.0](LICENSE). 기존 RecSim 저작권 고지를 보존합니다.
This is not an officially supported Google product.
