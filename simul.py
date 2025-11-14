import json
import numpy as np
from tqdm import tqdm

from recsim.environments.interest_evolution import create_environment


def convert_doc_obs(doc_obs):
    # OrderedDict -> dict of lists
    return {int(k): v.tolist() for k, v in doc_obs.items()}


def run_single_episode(env, steps=100):
    obs = env.reset()

    slate_size = env._environment.slate_size
    num_candidates = env._environment.num_candidates

    episode_data = []

    for t in range(steps):
        # action: 셔플된 slate_size개 문서 인덱스
        action = np.random.choice(num_candidates, size=slate_size, replace=False)

        # Gymnasium: step() returns 5 values
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # user observation is ndarray
        if isinstance(obs["user"], np.ndarray):
            user_obs = obs["user"].tolist()
        else:
            user_obs = obs["user"]

        # doc observation is OrderedDict
        doc_obs = convert_doc_obs(obs["doc"])

        episode_data.append({
            "step": t,
            "action": action.tolist(),
            "user": user_obs,
            "doc": doc_obs,
            "response": obs["response"],  # response는 이미 dict 리스트
            "reward": reward
        })

        if done:
            break

    return episode_data


def simulate_users(num_users=100, steps=100, output_file="sim_output.jsonl"):
    """
    100명 × 100 step 시뮬레이션을 jsonl로 저장
    """

    env_config = {
        "slate_size": 5,
        "num_candidates": 20,
        "resample_documents": True,
        "seed": 1,
    }

    env = create_environment(env_config)

    with open(output_file, "w", encoding="utf-8") as f:
        for user_id in tqdm(range(num_users), desc="Simulating Users"):
            episode = run_single_episode(env, steps=steps)

            f.write(
                json.dumps({
                    "user_id": user_id,
                    "steps": episode
                }, ensure_ascii=False) + "\n"
            )

    print(f"저장 완료: {output_file}")


if __name__ == "__main__":
    simulate_users()
