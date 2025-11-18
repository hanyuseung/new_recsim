# test_simul.py

from simulate_api import simulate_users_csv
from simulate_api import simulate_users_json

if __name__ == "__main__":
    simulate_users_json(
        slate_size=5,
        num_candidates=20,
        num_users=10,
        steps=10,
        file_name="data2.jsonl",
        global_seed=42,
        sim_seed=1,
    )
    simulate_users_csv(
        slate_size=5,
        num_candidates=20,
        num_users=10,
        steps=10,
        file_name="data1.csv",
        global_seed=42,
        sim_seed=1,
    )
