# test_simul.py

from simulate_api import simulate_users_csv
from simulate_api import simulate_users_json

if __name__ == "__main__":
    # simulate_users_json(
    #     slate_size=5,
    #     num_candidates=20,
    #     num_users=20,
    #     steps=10,
    #     file_name="data1.jsonl",
    #     global_seed=42, #이게 랜덤
    #     sim_seed=1, # enviroment
    # )
    simulate_users_csv(
        slate_size=5,
        num_candidates=20, # item 개수
        num_users=20,
        steps=10,
        file_name="data1.csv",
        global_seed=42,
        sim_seed=1,
    )
