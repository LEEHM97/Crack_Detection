import os


def labelme_json_to_dataset(json_path):
    os.system(
        "labelme_json_to_dataset " + json_path + " -o " + json_path.replace(".", "_")
    )


json_path = "C:\\Users\\gmlkd\\기업연계프로젝트\\test.json"

labelme_json_to_dataset(json_path)