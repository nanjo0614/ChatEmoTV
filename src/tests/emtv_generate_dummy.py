"""
EmoMusicTV ダミー生成スクリプト
- ランダムではなく固定コード (C major triad) を 8 小節出力
- miditoolkit の TempoChange オブジェクトを正しく追加
使い方:
    python src/tests/emtv_generate_dummy.py \
        --bars 8 --output data/demos/random_8bars.mid
"""

import argparse
from pathlib import Path

from miditoolkit import MidiFile, Instrument, Note, TempoChange


def make_dummy_midi(bars: int = 8, tempo: int = 120) -> MidiFile:
    """
    bars 小節分の簡単なコード進行 (全音符) を持つ MIDI を返す
    """
    midi = MidiFile()
    midi.ticks_per_beat = 480

    # ◆ テンポイベント (重要) ------------------------------------------
    # miditoolkit は start_time=0 の TempoChange が必須
    midi.tempo_changes.append(TempoChange(tempo=tempo, time=0))

    # ◆ トラック (Piano) ------------------------------------------------
    piano = Instrument(program=0, is_drum=False, name="Piano")

    ticks_per_bar = midi.ticks_per_beat * 4  # 4/4 拍想定
    for bar in range(bars):
        start_tick = bar * ticks_per_bar
        end_tick = start_tick + ticks_per_bar

        # C メジャー (C4, E4, G4)
        for pitch in (60, 64, 67):
            piano.notes.append(
                Note(
                    velocity=90,
                    pitch=pitch,
                    start=start_tick,
                    end=end_tick,
                )
            )

    midi.instruments.append(piano)
    return midi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=8, help="小節数 (4/4 拍)")
    ap.add_argument("--tempo", type=int, default=120, help="BPM")
    ap.add_argument("--output", required=True, help="保存先 .mid")
    args = ap.parse_args()

    midi = make_dummy_midi(bars=args.bars, tempo=args.tempo)

    # 出力フォルダを自動生成
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    midi.dump(args.output)
    print("✅ 生成完了:", args.output)


if __name__ == "__main__":
    main()
