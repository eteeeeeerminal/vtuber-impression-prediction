# 謝金まわりで必要なドキュメントを作成
from process_raw_data.data_types.shaped import load_shaped_datum
from process_raw_data.clerical import output_reward_data, output_email_txt

shaped_datum = load_shaped_datum("./data/12-08/vtuber-onomatopoeia.json")
output_reward_data(shaped_datum, "./data/reward.tsv")
output_email_txt(shaped_datum, "./data/reward_email.txt")
