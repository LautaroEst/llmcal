

import yaml

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def main():
    data = load_yaml("configs/task/tony_zhao_agnews_qa.yaml")
    print(data["casting"]["question"])
    print(data["casting"]["label2answer"])

if __name__ == "__main__":
    main()