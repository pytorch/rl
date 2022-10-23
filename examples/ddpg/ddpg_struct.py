
if __name__ == "__main__":
    GlobalHydra.instance().clear()
    hydra.initialize("../configs/")
    request.addfinalizer(GlobalHydra.instance().clear)

    cfg = hydra.compose("ddpg")

