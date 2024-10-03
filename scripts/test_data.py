import utils
from pprint import pprint

version = 31


def run_all():

    nvd_data = utils.read_data(f"../data/nvd_{version}_cleaned.pkl")
    data = nvd_data["data"]
    pprint(data[0])


run_all()
