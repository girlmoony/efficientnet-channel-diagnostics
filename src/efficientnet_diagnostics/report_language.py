"""Japanese HTML text, retaining English plot labels and artifact filenames."""
import re


_TRANSLATIONS = (
    ('<html lang="en">', '<html lang="ja">'),
    ('<title>EfficientNet Channel Diagnostics</title>', '<title>EfficientNet チャンネル診断</title>'),
    ('<h1>Channel contributions and spatial diagnostics</h1>', '<h1>チャンネル寄与と画像領域の診断</h1>'),
    ('<h2>Hot versus low-contribution region controls</h2>', '<h2>ホット領域と低貢献領域の比較</h2>'),
    ('A (true class) and B (comparison class) remain fixed from the original input, even if Top-1 changes. Positive margin_drop means B-A decreased. Positive hot_minus_low_drop means hotspot replacement reduced B-A more than the control; a single result does not establish causality.',
     '正解クラスAと比較クラスBは元の画像で選択したクラスに固定します。置換後にTop-1が変わっても同じA/Bで比較します。正の margin_drop はB-Aの減少、正の hot_minus_low_drop はホット領域の置換による減少が対照領域より大きいことを示します。単一の結果だけでは因果関係を確定できません。'),
    ('Full experiment JSON', '実験全体のJSON'),
    ('<th>Target/method/region</th>', '<th>対象/手法/領域</th>'),
    ('<th>Area fraction</th>', '<th>面積比率</th>'),
    ('<th>Margin drop</th>', '<th>低下量</th>'),
    ('<th>New Top-1</th>', '<th>新しいTop-1</th>'),
    ('Correct prediction.', '正解。'),
    ('Incorrect prediction.', '誤認識。'),
    ('True class A:', '正解クラスA:'),
    ('predicted Top-1:', '予測Top-1:'),
    ('comparison class B (runner-up):', '比較クラスB（第2候補）:'),
    ('comparison class B (incorrect Top-1):', '比較クラスB（誤認識Top-1）:'),
    ('B-A logit margin:', 'B-Aのlogit差:'),
    ('bias difference:', 'bias差:'),
    ('reconstruction error:', '再構成誤差:'),
    ('Red pushes B; blue supports A. Shares are computed within each direction,\nexclude bias, and are not probabilities. Channel overlays share a color scale; the total map\nuses its own scale. Native feature map size:',
     '赤はBへの寄与、青はAへの寄与です。占比は方向ごとに計算し、biasを含まず、確率ではありません。チャンネルのオーバーレイは共通の色スケールを使い、全体マップは独自のスケールを使います。元の特徴マップサイズ:'),
    ('; interpolation is for display only.\nSpatial positions describe feature responses, not precise pixel-level causal attribution.\nBackground explanations require controlled image interventions.',
     '。補間は表示のためだけに使用します。空間位置は特徴の応答を表し、厳密な画素単位の因果帰属ではありません。背景の影響を説明するには制御した画像置換実験が必要です。'),
    ('All channel values (CSV)', '全チャンネルの値（CSV）'),
    ('Scores and run configuration (JSON)', 'スコアと実行設定（JSON）'),
    ('Native spatial contributions (NPZ)', '元の解像度の空間寄与（NPZ）'),
    ('All channel spatial concentration (CSV)', '全チャンネルの空間集中度（CSV）'),
    ('Spatial metrics (JSON)', '空間指標（JSON）'),
    ('Metrics are computed separately for positive B contributions and absolute negative A contributions. Entropy is normalized by log(total grid cells); concentration shares and entropy are null when there is no contribution. Area coverage is the fraction of strictly positive grid cells, not object segmentation area. Detail panels use independent scales for each channel; compare numeric values.',
     '指標はBへの正の寄与とAへの負の寄与の絶対値について別々に計算します。エントロピーは全グリッド数の対数で正規化し、寄与がない場合は集中度の占比とエントロピーをnullとします。面積比率は正の値を持つグリッドの割合で、物体の領域分割ではありません。詳細パネルのスケールはチャンネルごとに異なるため、数値で比較してください。'),
)


def japanese_page(page):
    # Preserve user-provided labels verbatim, including names resembling UI text.
    parts = re.split(r'(<b>.*?</b>)', page, flags=re.DOTALL)
    for index in range(0, len(parts), 2):
        for source, translated in _TRANSLATIONS:
            parts[index] = parts[index].replace(source, translated)
    return ''.join(parts)
