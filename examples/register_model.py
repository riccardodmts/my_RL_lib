if __name__ == "__main__":
    # import the module: in this way all the register_model calls are executed
    from utils import model_catalog

    model_dict = {

        "input_dim": 10,
        "num_actions": 4,

        "actor": {
            "first_hidden": 32,
            "second_hidden": 32
        },

        "critic": {

            "first_hidden": 32

        }
    }

    instance = model_catalog.import_model("example", model_dict)
    print(instance)