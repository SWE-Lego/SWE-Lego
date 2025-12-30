import pyarrow.parquet as pq
import json
import argparse


def convert_parquet_to_json(data_name):
    table = pq.read_table(f'{data_name}.parquet')
    df = table.to_pandas()

    json_data = df.apply(lambda row: {
        **{col: row[col].tolist() if hasattr(row[col], 'tolist') else row[col] 
        for col in df.columns}
    }, axis=1).tolist()

    with open(f"{data_name}.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    print(f"Convert parquet to json and save to {data_name}.json")


def repalce_newline_in_json(data_name):
    with open(f"{data_name}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        # replace \r\n by \n for every message in the "messages" field for each item, make sure the change is made in place
        for item in data:
            for message in item["messages"]:
                while "\r\n" in message["content"]:
                    # replace \r\n with \n
                    message["content"] = message["content"].replace("\r\n", "\n")
            
            # check if replace was successful
            for message in item["messages"]:
                assert "\r\n" not in message["content"], "Replacement failed for item"

    with open(f"{data_name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Repalce newline in json and save to {data_name}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default="swe_smith_train_5k_original", type=str)
    args = parser.parse_args()

    convert_parquet_to_json(args.data_name)
    repalce_newline_in_json(args.data_name)