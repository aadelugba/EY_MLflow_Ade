from src.train_model import train_model, get_run_id

params = {"max_features": 0.7, "n_estimators": 100, "max_depth" : 10}

if __name__ == "__main__":
    print("Training Model & Extracting Loss Function ...")
    print(
        train_model(params)
    )
    print("Get Run Id")
    # print(
    #     get_run_id(train_model)
    #     )