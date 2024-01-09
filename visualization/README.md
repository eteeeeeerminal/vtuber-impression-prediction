# annotation-stat
アノテーションデータに対していろんな統計的分析・可視化を行う

mecab と poetry を入れて
mecab はパスが通ってることを確認すること(pythonのパッケージとは別に入れる)
mecab: https://taku910.github.io/mecab/
これもやる
python -m unidic download
https://analytics-note.xyz/programming/python-mecabrc-error/

IPAexフォント: https://moji.or.jp/ipafont/ipaex00401/


# 実行ログメモ
- img_single: senet 最終層のみ更新
  - 03ea23f3, f4c6910f
  - 9, 6, 0, 3, 4
  - 9: lr0.0001 dropout0.5 wd1.0
- img_all: senet 全層更新
  - 637fc1dd, 637fc1dd
  - 10, 6, 9, 11, 5
  - 10: lr0.0001 dropout0.5 wd0.001
- mfcc
  - b1dd8ab5
  - 17, 16, 23, 15, 22
  - 17: lr1e-05 wd20.0
- logfbank
  - 36a9a96f
  - 23, 17, 22, 16, 21
  - 23: lr5e-06 wd20.0
