# New RecSim (Refactored for Python3 / Gymnasium / PyTorch Workflow)

**New RecSim**은 Google RecSim(2019)을 기반으로 하되,  
_사용자–아이템 RandomWalk 기반 시뮬레이션을 간결하고 현대적인 환경에서 재구축한 버전입니다._

본 리팩토링은 추천시스템 연구·시뮬레이션을 **PyTorch·Gymnasium·Python3** 환경에서  
바로 사용할 수 있도록 하기 위해 수행되었습니다.

---

## 1. 주요 변경 사항

### ✔ Python 3 전용
- six / future 제거
- modern typing 활용 가능
- 코드 전체 Python3 문법으로 정리

### ✔ Gym → Gymnasium 기반 Wrapper
- `recsim_gym.RecSimGymEnv`를 gymnasium API로 재작성  
  → RL agent / simulation tool과 호환성 상승

### ✔ RandomWalk 기반 상호작용 시뮬레이션 강화
- 사용자 관심도(user_interests) + 아이템 features 기반 dot-product scoring
- MNL / Proportional / Cascade Choice Model modern implementation
- User state update functions 리팩토링

### ✔ PyTorch 친화적 구조
- 내부는 numpy 기반 유지  
- 외부에서 torch.tensor 변환이 매우 쉬움
- 시뮬레이션 데이터를 바로 PyTorch 학습데이터로 만들기 용이

### ✔ 최신 NumPy(2.x) 호환성 확보
- numpy RandomState 유지
- np.random.Generator와 충돌 제거
- randint → integers 로 수정

### ✔ Legacy RecSim 구조 유지
- document / user / choice_model / environment 전체 구조는 동일  
- 단, modern 환경에서 동작하도록 전면 정리

---

## 2. 수정된 주요 파일

다음 모듈들이 완전 리팩토링 되었습니다:

| 파일명 | 설명 |
|-------|------|
| **document.py** | AbstractDocument / CandidateSet modern 정리 |
| **user.py** | User state / sampler / response 구조 개선 |
| **choice_model.py** | MNL / Proportional / Cascade modern re-implementation |
| **environment.py** | SingleUser / MultiUser 환경 정리 |
| **recsim_gym.py** | Gymnasium Env Wrapper 신규 작성 |
| **interest_evolution.py** | 전체 환경 종합: DocumentSampler, UserModel, Reward defined |
| simulate_api.py | json 혹은 csv 로 randomwalk 시뮬레이션 해주는 API |
| test_simul.py | csv로 10*20 시뮬레이션 결과를 출력해준다. |

---

## 3. 설치 방법

### 3.1 의존성 설치

```bash
pip install recsim numpy tqdm torch gymnasium
```
### 3.2 예제 실행
```
python3 simulate_api.py
```

## 4. 예제 출력
```csv
user_id,step,user_0,user_1,user_2,user_3,user_4,user_5,user_6,user_7,user_8,user_9,user_10,user_11,user_12,user_13,user_14,user_15,user_16,user_17,user_18,user_19,response_click,response_watch_time,response_liked,response_quality,response_cluster_id,doc_0_feat_0,doc_0_feat_1,doc_0_feat_2,doc_0_feat_3,...,doc_19_feat_19
0,0,0.6491,0.9365,-0.3731,0.3846,0.7527,0.7892,-0.8299,-0.9218,-0.6603,0.7562,-0.8033,-0.1577,0.9157,0.0663,0.3837,-0.3689,0.3730,0.6692,-0.9634,0.5002,0,0.0,0,1.0,3,0,0,0,0,...,0

```
컬럼 의미

| 컬럼명                   | 설명                            |
| --------------------- | ----------------------------- |
| `user_id`             | 시뮬레이션된 유저 번호                  |
| `step`                | 해당 유저의 t-step                 |
| `user_0` ~ `user_19`  | 유저 상태 벡터 (20차원)               |
| `response_click`      | 클릭 여부 (0/1)                   |
| `response_watch_time` | 시청 시간                         |
| `response_liked`      | 좋아요 여부                        |
| `response_quality`    | 문서 품질                         |
| `response_cluster_id` | 문서 클러스터 ID                    |
| `doc_X_feat_Y`        | 문서 X의 feature vector 중 Y번째 요소 |

doc_X_feat_Y는 원하는 feature기반 시뮬레이션을 할때 사용됩니다.  
하지만 randomwalk기반의 추천 시뮬레이션에서 이는 필요하지 않기 때문에,  
모든 doc_X_feat_Y는 0값을 갖습니다. 

이 구조는 바로 다음 목적에 사용할 수 있습니다:

추천 모델 학습용 트레이닝 데이터 생성

행동 정책 평가 (offline RL / simulation)

user preference random-walk 분석

cluster / category 별 CTR 통계 분석

## 5. API 사용법 ( simulate_user_csv/json() )

### 사용 예시

```python
simulate_users_csv(
    slate_size=5,
    num_candidates=20,
    num_users=100,
    steps=100,
    file_name="sim_output.csv",
    global_seed=42,
    sim_seed=1,
)

simulate_users_json(
    slate_size=5,
    num_candidates=20,
    num_users=100,
    steps=100,
    file_name="sim_output.json",
    global_seed=42,
    sim_seed=1,
)
```

### 파라미터 상세
| 파라미터             | 타입  | 기본값              | 설명                                              |
| ---------------- | --- | ---------------- | ----------------------------------------------- |
| `slate_size`     | int | 5                | 매 step에서 추천할 문서 개수(K). Top-K recommendation에 해당 |
| `num_candidates` | int | 20               | 한 타임스텝에서 사용자에게 보여줄 수 있는 전체 후보 아이템 수             |
| `num_users`      | int | 100              | 시뮬레이션할 유저 수                                     |
| `steps`          | int | 100              | 각 유저가 수행할 step 수 (interaction sequence 길이)      |
| `file_name`      | str | "sim_output.csv" | 저장될 CSV 파일 이름                                   |
| `global_seed`    | int | 42               | 외부(random action 등)에 사용되는 RNG seed              |
| `sim_seed`       | int | 1                | RecSim 내부(document/user/choice model) RNG seed  |


## 6. Flow chart
```java
UserSampler → UserState
                 ↓
DocumentSampler → CandidateSet
                 ↓
ChoiceModel → Selected Action
                 ↓
ResponseModel → Click/Watch/Liked
                 ↓
UserState Update (RandomWalk)

```

# RecSim: A Configurable Recommender Systems Simulation Platform

RecSim is a configurable platform for authoring simulation environments for
recommender systems (RSs) that naturally supports **sequential interaction**
with users. RecSim allows the creation of new environments that reflect
particular aspects of user behavior and item structure at a level of abstraction
well-suited to pushing the limits of current reinforcement learning (RL) and RS
techniques in sequential interactive recommendation problems. Environments can
be easily configured that vary assumptions about: user preferences and item
familiarity; user latent state and its dynamics; and choice models and other
user response behavior. We outline how RecSim offers value to RL and RS
researchers and practitioners, and how it can serve as a vehicle for
academic-industrial collaboration. For a detailed description of the RecSim
architecture please read [Ie et al](https://arxiv.org/abs/1909.04847). Please
cite the paper if you use the code from this repository in your work.

### Bibtex

```
@article{ie2019recsim,
    title={RecSim: A Configurable Simulation Platform for Recommender Systems},
    author={Eugene Ie and Chih-wei Hsu and Martin Mladenov and Vihan Jain and Sanmit Narvekar and Jing Wang and Rui Wu and Craig Boutilier},
    year={2019},
    eprint={1909.04847},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

<a id='Disclaimer'></a>

## Disclaimer

This is not an officially supported Google product.

## What's new

*   **12/13/2019:** Added (abstract) classes for both multi-user environments
    and agents. Added bandit algorithms for generalized linear models.

## Installation and Sample Usage

It is recommended to install RecSim using (https://pypi.org/project/recsim/):

```shell
pip install recsim
```

However, the latest version of Dopamine is not in PyPI as of December, 2019. We
want to install the latest version from Dopamine's repository like the following
before we install RecSim. Note that Dopamine requires Tensorflow 1.15.0 which is
the final 1.x release including GPU support for Ubuntu and Windows.

```
pip install git+https://github.com/google/dopamine.git
```

Here are some sample commands you could use for testing the installation:

```
git clone https://github.com/google-research/recsim
cd recsim/recsim
python main.py --logtostderr \
  --base_dir="/tmp/recsim/interest_exploration_full_slate_q" \
  --agent_name=full_slate_q \
  --environment_name=interest_exploration \
  --episode_log_file='episode_logs.tfrecord' \
  --gin_bindings=simulator.runner_lib.Runner.max_steps_per_episode=100 \
  --gin_bindings=simulator.runner_lib.TrainRunner.num_iterations=10 \
  --gin_bindings=simulator.runner_lib.TrainRunner.max_training_steps=100 \
  --gin_bindings=simulator.runner_lib.EvalRunner.max_eval_episodes=5
```

You could then start a tensorboard and view the output

```
tensorboard --logdir=/tmp/recsim/interest_exploration_full_slate_q/ --port=2222
```

You could also find the simulated logs in /tmp/recsim/episode_logs.tfrecord

## Tutorials

To get started, please check out our Colab tutorials. In
[**RecSim: Overview**](recsim/colab/RecSim_Overview.ipynb),
we give a brief overview about RecSim. We then talk about each configurable
component:
[**environment**](recsim/colab/RecSim_Developing_an_Environment.ipynb)
and
[**recommender agent**](recsim/colab/RecSim_Developing_an_Agent.ipynb).

## Documentation


Please refer to the [white paper](http://arxiv.org/abs/1909.04847) for the
high-level design.


