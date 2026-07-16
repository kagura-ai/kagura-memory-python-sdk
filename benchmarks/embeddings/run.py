#!/usr/bin/env python3
"""Benchmark: Ollama vs OpenAI embeddings across 20 domain contexts.

Tests remember/recall quality and performance with 8 query types.
Outputs Rich terminal tables + Markdown report for publishing.

Usage:
    uv run python examples/benchmark_embeddings.py
    uv run python examples/benchmark_embeddings.py --cleanup
"""

import asyncio
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from rich.console import Console
from rich.table import Table

from kagura_memory import KaguraClient
from kagura_memory.config import load_config

console = Console()

# ---------------------------------------------------------------------------
# Query types
# ---------------------------------------------------------------------------
# Each query has:
#   query: search text
#   type: exact|synonym|concept|question|vague|cross|negative|multi-hop
#   expected_idx: index into domain memories that should match (None for negative)
#   difficulty: easy|medium|hard|control

EMBEDDING_MODELS = [
    "qwen3-embedding:8b",
]

# ---------------------------------------------------------------------------
# Domain definitions — 20 domains, each with memories + typed queries
# ---------------------------------------------------------------------------

DOMAINS: list[dict] = [
    {
        "name": "education",
        "display_name": "Education",
        "summary": "School administration, curriculum design, student management, EdTech",
        "memories": [
            {
                "summary": "GIGAスクール構想: 1人1台端末とクラウド活用で個別最適化学習を実現",
                "content": "文科省GIGAスクール構想により全国の小中学校で1人1台端末が配備。Google Workspace for EducationやMicrosoft 365 Educationが主要プラットフォーム。校務DXとして統合型校務支援システム（C4th等）の導入が進む。",
                "type": "note",
                "tags": ["education", "giga-school", "edtech"],
                "importance": 0.8,
            },
            {
                "summary": "不登校児童生徒へのICT活用支援: オンライン出席認定の要件と運用",
                "content": "2023年通知により不登校児童のオンライン学習が出席扱い可能に。要件: 担任との定期面談、学習計画の提出、ICTを活用した学習記録。Classi、スタディサプリ等のLMS活用が拡大。",
                "type": "decision",
                "tags": ["education", "truancy", "online-learning"],
                "importance": 0.7,
            },
            {
                "summary": "探究学習の評価手法: ルーブリック評価とポートフォリオ評価の併用",
                "content": "高校の総合的な探究の時間では、プロセス評価が重要。ルーブリック（4段階）で思考力・表現力を評価し、ポートフォリオで成長過程を可視化。Google Sitesやロイロノートで電子ポートフォリオを運用する学校が増加。",
                "type": "learning",
                "tags": ["education", "assessment", "inquiry-learning"],
                "importance": 0.6,
            },
        ],
        "queries": [
            {
                "query": "GIGAスクール端末の活用方法",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "学校のタブレット配備とクラウドサービス",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "教育のデジタル化推進施策",
                "type": "concept",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "不登校の子供がオンラインで学ぶにはどうすればいい？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "学校行けない子の対応って何かある？",
                "type": "vague",
                "expected_idx": 1,
                "difficulty": "hard",
            },
            {
                "query": "保育園でのICT活用について",
                "type": "cross",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "量子コンピュータの教育利用",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "不登校の子のオンライン探究学習を評価する方法",
                "type": "multi-hop",
                "expected_idx": [1, 2],
                "difficulty": "hard",
            },
        ],
    },
    {
        "name": "welfare",
        "display_name": "Welfare & Social Services",
        "summary": "Social welfare, elderly care, disability support, community services",
        "memories": [
            {
                "summary": "介護保険制度の区分変更申請: 要介護度が実態と合わない場合の対処法",
                "content": "要介護認定の結果に不服がある場合、区分変更申請が可能。主治医意見書の内容が重要で、ADL低下の具体的記載を依頼する。認定調査時の特記事項に日常の困りごとを詳細に伝えることがポイント。",
                "type": "note",
                "tags": ["welfare", "care-insurance", "elderly"],
                "importance": 0.8,
            },
            {
                "summary": "障害者総合支援法の就労支援: A型・B型事業所の違いと選択基準",
                "content": "就労継続支援A型は雇用契約あり（最低賃金保障）、B型は雇用契約なし（工賃支払い）。A型は週20時間以上の就労が可能な方、B型は体調に波がある方や短時間から始めたい方に適する。",
                "type": "decision",
                "tags": ["welfare", "disability", "employment-support"],
                "importance": 0.7,
            },
            {
                "summary": "地域包括支援センターの役割: 高齢者の総合相談と介護予防マネジメント",
                "content": "地域包括支援センターは高齢者の相談窓口。保健師・社会福祉士・主任ケアマネの3職種配置。介護予防ケアプラン作成、権利擁護（虐待対応・成年後見）、包括的・継続的ケアマネジメント支援を実施。",
                "type": "note",
                "tags": ["welfare", "community-care", "elderly"],
                "importance": 0.7,
            },
        ],
        "queries": [
            {
                "query": "介護認定の変更手続き",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "要介護度に納得できないときの不服申立て",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "障害者の社会参加支援制度",
                "type": "concept",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "障害があっても働ける場所ってどんなのがある？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "お年寄りの困りごとどこに相談する？",
                "type": "vague",
                "expected_idx": 2,
                "difficulty": "hard",
            },
            {
                "query": "医療費の助成制度",
                "type": "cross",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "外国人労働者の在留資格",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "介護認定に不服がある高齢者の相談先と変更手続き",
                "type": "multi-hop",
                "expected_idx": [0, 2],
                "difficulty": "hard",
            },
        ],
    },
    {
        "name": "healthcare",
        "display_name": "Healthcare & Medical",
        "summary": "Medical practice, clinical decisions, health informatics",
        "memories": [
            {
                "summary": "電子カルテのHL7 FHIR対応: 医療情報の標準化と相互運用性の実現",
                "content": "HL7 FHIRはRESTful APIベースの医療情報交換標準。Patient、Observation、MedicationRequestなどのリソース型でデータを構造化。日本では厚労省の医療DX推進にてFHIR JP Coreプロファイルが策定中。",
                "type": "learning",
                "tags": ["healthcare", "fhir", "interoperability"],
                "importance": 0.8,
            },
            {
                "summary": "敗血症の早期発見: qSOFAスコアとSIRS基準の使い分け",
                "content": "qSOFA（呼吸数≥22、意識変容、収縮期血圧≤100mmHg）はICU外でのスクリーニングに有用。SIRS基準より特異度が高い。2項目以上で敗血症を疑い、血液培養とラクテート測定を実施。",
                "type": "note",
                "tags": ["healthcare", "sepsis", "emergency"],
                "importance": 0.9,
            },
            {
                "summary": "ポリファーマシー対策: 高齢者の多剤併用リスクと処方適正化の手順",
                "content": "6剤以上でポリファーマシー。高齢者の薬物有害事象リスクが増大。処方適正化にはSTOPP/STARTクライテリアを活用。かかりつけ薬剤師による一元管理、お薬手帳の活用、処方カスケードの回避が重要。",
                "type": "decision",
                "tags": ["healthcare", "polypharmacy", "elderly-care"],
                "importance": 0.7,
            },
        ],
        "queries": [
            {
                "query": "FHIRによる医療データ連携",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "病院間の診療情報共有の国際標準",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "医療のデジタルトランスフォーメーション",
                "type": "concept",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "救急でsepsisを見逃さないためのスコアリングは？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "おばあちゃんの薬が多すぎる気がする",
                "type": "vague",
                "expected_idx": 2,
                "difficulty": "hard",
            },
            {
                "query": "AI画像診断の保険適用",
                "type": "cross",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "遺伝子治療の倫理問題",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "高齢患者のqSOFA評価と多剤併用の関連",
                "type": "multi-hop",
                "expected_idx": [1, 2],
                "difficulty": "hard",
            },
        ],
    },
    {
        "name": "childcare",
        "display_name": "Childcare & Early Education",
        "summary": "Nursery management, child development, parenting support",
        "memories": [
            {
                "summary": "保育ICT化: 午睡チェックセンサーとヒヤリハット管理システムの導入効果",
                "content": "午睡中のSIDS予防としてマット型センサー（ルクミー、ベビーセンスなど）を導入。5分間隔の目視確認を補完し、体動停止時にアラート。ヒヤリハット報告のデジタル化により分析・再発防止が容易に。",
                "type": "note",
                "tags": ["childcare", "ict", "safety"],
                "importance": 0.8,
            },
            {
                "summary": "保育所における医療的ケア児の受入れ: 看護師配置と研修要件",
                "content": "医療的ケア児支援法（2021年施行）により保育所での受入れ体制整備が義務化。看護師または認定特定行為業務従事者の配置が必要。喀痰吸引、経管栄養等の対応について個別マニュアルを作成。",
                "type": "decision",
                "tags": ["childcare", "medical-care", "inclusion"],
                "importance": 0.7,
            },
            {
                "summary": "保育士の配置基準改善: 1歳児6対1から5対1への見直し議論",
                "content": "2024年度から4-5歳児の配置基準が30対1から25対1に改善。1歳児は現行6対1だが5対1への見直しを保育団体が要望。配置改善には財源確保と保育士確保が課題。処遇改善加算の拡充で人材確保を図る。",
                "type": "note",
                "tags": ["childcare", "staffing", "policy"],
                "importance": 0.7,
            },
        ],
        "queries": [
            {
                "query": "午睡チェックのICTセンサー",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "お昼寝中の赤ちゃん見守りテクノロジー",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "保育の安全管理と事故予防",
                "type": "concept",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "医療的ケアが必要な子を保育園に預けるには？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "保育園の先生足りなくない？",
                "type": "vague",
                "expected_idx": 2,
                "difficulty": "hard",
            },
            {
                "query": "幼児教育のカリキュラム設計",
                "type": "cross",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "ベビーシッターのマッチングアプリ",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "医療的ケア児受入れに伴う保育士配置基準の影響",
                "type": "multi-hop",
                "expected_idx": [1, 2],
                "difficulty": "hard",
            },
        ],
    },
    {
        "name": "finance",
        "display_name": "Finance & Banking",
        "summary": "Banking, fintech, regulatory compliance, risk management",
        "memories": [
            {
                "summary": "バーゼルIII最終化: 信用リスクの標準的手法における外部格付使用の制限",
                "content": "バーゼルIII最終規則では、信用リスクの標準的手法で外部格付への過度な依存を抑制。デューデリジェンス要件を強化し、格付が利用できない場合のリスクウェイト（100%〜150%）を規定。2028年完全適用。",
                "type": "note",
                "tags": ["finance", "basel3", "credit-risk"],
                "importance": 0.9,
            },
            {
                "summary": "即時決済システム「ことら」: 全銀システムとの違いと10万円以下送金の利点",
                "content": "ことらは少額送金（10万円以下）に特化した低コスト即時送金システム。銀行口座番号の代わりに携帯番号やメールアドレスで送金可能。全銀システムの手数料（数百円）に対し、ことらは無料〜数十円。",
                "type": "learning",
                "tags": ["finance", "payment", "fintech"],
                "importance": 0.7,
            },
            {
                "summary": "AML/CFT対応: 継続的顧客管理(CDD)とリスクベースアプローチの実装",
                "content": "FATF第4次相互審査の指摘を受け、継続的顧客管理を強化。取引モニタリングシステムでリスクスコアリング、PEPs（重要な公的地位の人物）の強化確認、制裁リストスクリーニングの自動化が必須。",
                "type": "note",
                "tags": ["finance", "aml", "compliance"],
                "importance": 0.8,
            },
        ],
        "queries": [
            {
                "query": "バーゼル規制の信用リスク計算",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "銀行の自己資本比率規制の最新動向",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "金融規制とリスク管理",
                "type": "concept",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "友達に安く送金できるサービスって何？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "お金の不正利用を防ぐ仕組み",
                "type": "vague",
                "expected_idx": 2,
                "difficulty": "hard",
            },
            {
                "query": "暗号資産取引所のセキュリティ",
                "type": "cross",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "為替ヘッジの手法とコスト計算",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "ことら送金のAML対応要件",
                "type": "multi-hop",
                "expected_idx": [1, 2],
                "difficulty": "hard",
            },
        ],
    },
    {
        "name": "science",
        "display_name": "Science & Research",
        "summary": "Scientific research, lab management, academic publishing",
        "memories": [
            {
                "summary": "CRISPRオフターゲット検出: GUIDE-seqとCIRCLE-seqの比較と選択基準",
                "content": "GUIDE-seqは細胞内でのオフターゲット切断部位を同定（in vivo）。CIRCLE-seqはゲノムDNAをin vitroで環状化して切断部位を検出（高感度だが偽陽性多い）。臨床応用にはGUIDE-seq推奨、スクリーニングにはCIRCLE-seq。",
                "type": "learning",
                "tags": ["science", "crispr", "genome-editing"],
                "importance": 0.8,
            },
            {
                "summary": "研究データ管理計画(DMP): JSTとAMEDのオープンデータ方針の違い",
                "content": "JST: 原則公開、猶予期間は論文発表後2年。AMED: 医療データは個人情報保護の観点から制限付き公開。両機関ともDMP提出が義務化。メタデータはJPCOARスキーマで記述し、機関リポジトリに登録。",
                "type": "decision",
                "tags": ["science", "open-data", "research-management"],
                "importance": 0.7,
            },
        ],
        "queries": [
            {
                "query": "CRISPRのオフターゲット評価方法",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "ゲノム編集の安全性検証技術",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "研究の再現性と透明性",
                "type": "concept",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "科研費の成果データはどこまで公開しなきゃいけない？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "論文のデータ公開ルール",
                "type": "vague",
                "expected_idx": 1,
                "difficulty": "hard",
            },
            {
                "query": "タンパク質構造予測",
                "type": "cross",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "核融合炉の材料科学",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "CRISPR研究のデータ公開義務と安全性評価",
                "type": "multi-hop",
                "expected_idx": [0, 1],
                "difficulty": "hard",
            },
        ],
    },
    {
        "name": "construction",
        "display_name": "Construction & Civil Engineering",
        "summary": "Building construction, civil engineering, BIM, safety management",
        "memories": [
            {
                "summary": "BIM/CIM活用: i-Constructionにおける3次元モデルの活用段階と効果",
                "content": "国交省のi-Constructionでは段階的BIM/CIM活用を推進。フェーズ1: 設計段階の3D可視化、フェーズ2: 施工段階のフロントローディング、フェーズ3: 維持管理との連携。Revit、Civil 3D、InfraWorksが主要ツール。",
                "type": "note",
                "tags": ["construction", "bim", "i-construction"],
                "importance": 0.8,
            },
            {
                "summary": "コンクリート打設時の温度ひび割れ対策: マスコン判定と温度応力解析",
                "content": "部材最小寸法80cm以上でマスコンクリートと判定。温度応力解析（JCMAC等）でひび割れ指数を算出。指数1.0未満で対策必要。低熱ポルトランドセメント使用、パイプクーリング、打設リフト高さ制限が有効。",
                "type": "learning",
                "tags": ["construction", "concrete", "crack-control"],
                "importance": 0.7,
            },
        ],
        "queries": [
            {
                "query": "BIM活用の段階的導入方法",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "建設現場の3Dモデリング技術",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "建設業の生産性革命",
                "type": "concept",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "大きなコンクリート構造物にヒビが入らないようにするには？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "工事の品質管理",
                "type": "vague",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "橋梁の耐震補強工法",
                "type": "cross",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "木造住宅の断熱工法",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "BIMで温度ひび割れリスクを事前検証する方法",
                "type": "multi-hop",
                "expected_idx": [0, 1],
                "difficulty": "hard",
            },
        ],
    },
    {
        "name": "aerospace",
        "display_name": "Aerospace & Space",
        "summary": "Satellite operations, rocket engineering, space exploration",
        "memories": [
            {
                "summary": "小型SAR衛星コンステレーション: 合成開口レーダーによる全天候地表観測",
                "content": "SAR衛星は雲や夜間でも地表観測が可能。Synspective（日本）やICEYE（フィンランド）が小型SAR衛星コンステレーションを構築。分解能1m級、リビジット時間数時間を目指す。災害監視、インフラ変位検出に活用。",
                "type": "note",
                "tags": ["aerospace", "sar", "earth-observation"],
                "importance": 0.8,
            },
            {
                "summary": "デブリ除去技術: アストロスケールのELSA-dとRPO技術の実証成果",
                "content": "アストロスケールのELSA-d（2021年打上げ）はRPO（接近・近傍運用）技術を実証。磁石ドッキングプレートによるデブリ捕獲に成功。JAXA CRD2プログラムでは大型デブリ除去の商用化を2025年以降に計画。",
                "type": "learning",
                "tags": ["aerospace", "debris-removal", "orbital-mechanics"],
                "importance": 0.7,
            },
        ],
        "queries": [
            {
                "query": "SAR衛星による災害監視",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "曇りでも地表を撮影できる衛星技術",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "宇宙空間の持続可能性",
                "type": "concept",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "宇宙ゴミをどうやって片付けるの？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "衛星って何に使えるの",
                "type": "vague",
                "expected_idx": 0,
                "difficulty": "hard",
            },
            {
                "query": "ロケットエンジンの再利用技術",
                "type": "cross",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "火星テラフォーミング計画",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "SAR衛星でデブリ衝突リスクを監視",
                "type": "multi-hop",
                "expected_idx": [0, 1],
                "difficulty": "hard",
            },
        ],
    },
    {
        "name": "biotech",
        "display_name": "Biotechnology",
        "summary": "Biotech R&D, drug discovery, bioinformatics",
        "memories": [
            {
                "summary": "mRNA医薬品の非臨床安全性評価: LNP送達系の免疫原性とbiodistribution",
                "content": "mRNA-LNP製剤の非臨床評価では、LNP成分（イオン化脂質）による自然免疫活性化、肝臓への集積パターン、反復投与時の抗PEG抗体産生が重要な評価項目。FDAガイダンス（2023年）でbiodistribution試験の標準化が進む。",
                "type": "note",
                "tags": ["biotech", "mrna", "drug-safety"],
                "importance": 0.9,
            },
            {
                "summary": "AlphaFold2による構造予測のdrug discoveryへの応用と限界",
                "content": "AlphaFold2はタンパク質の静的構造予測に高精度だが、リガンド結合時のconformational changeは予測困難。分子動力学シミュレーション（GROMACS、Amber）との併用がドラッグデザインには必要。結合ポケットの精度はまだ不十分。",
                "type": "learning",
                "tags": ["biotech", "alphafold", "drug-discovery"],
                "importance": 0.8,
            },
        ],
        "queries": [
            {
                "query": "mRNAワクチンの安全性評価ポイント",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "脂質ナノ粒子製剤の毒性試験",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "AI創薬の最前線",
                "type": "concept",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "AlphaFoldで新薬って作れるの？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "タンパク質の形を予測するやつ",
                "type": "vague",
                "expected_idx": 1,
                "difficulty": "hard",
            },
            {
                "query": "ゲノム編集の安全性",
                "type": "cross",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "再生医療用のiPS細胞培養プロトコル",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "mRNA製剤のLNP体内分布をAlphaFoldで予測できるか",
                "type": "multi-hop",
                "expected_idx": [0, 1],
                "difficulty": "hard",
            },
        ],
    },
    {
        "name": "government",
        "display_name": "Government & Public Administration",
        "summary": "Digital government, policy making, public services",
        "memories": [
            {
                "summary": "デジタル庁ガバメントクラウド: AWS・GCP・Azure・OCI採択と移行計画",
                "content": "ガバメントクラウドとして4社（AWS、GCP、Azure、OCI）を採択。2025年度末までに地方自治体の基幹20業務をガバメントクラウドに移行予定。標準準拠システムへの移行費用は国が補助。共通基盤としてのID連携（GビズID）を推進。",
                "type": "decision",
                "tags": ["government", "cloud", "digital-transformation"],
                "importance": 0.9,
            },
            {
                "summary": "マイナンバーカードの電子証明書更新: J-LIS窓口対応の効率化手法",
                "content": "マイナカードの電子証明書（署名用・利用者証明用）は5年更新。更新はJ-LIS端末で実施、暗証番号再設定が必要。窓口混雑対策として予約制導入、コンビニでのオンライン更新（2025年導入）を検討中。",
                "type": "note",
                "tags": ["government", "mynumber", "digital-id"],
                "importance": 0.7,
            },
        ],
        "queries": [
            {
                "query": "ガバメントクラウドの移行スケジュール",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "自治体システムのクラウド統合",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {"query": "行政のDX推進", "type": "concept", "expected_idx": 0, "difficulty": "medium"},
            {
                "query": "マイナンバーカードの更新って何年ごと？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "役所の手続きがめんどくさい",
                "type": "vague",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "国のIT予算の使い方",
                "type": "cross",
                "expected_idx": 0,
                "difficulty": "hard",
            },
            {
                "query": "ふるさと納税の返礼品ルール",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "ガバメントクラウド上でマイナカード認証を統合する計画",
                "type": "multi-hop",
                "expected_idx": [0, 1],
                "difficulty": "hard",
            },
        ],
    },
    {
        "name": "legal",
        "display_name": "Legal & Compliance",
        "summary": "Legal practice, regulatory compliance, contract management",
        "memories": [
            {
                "summary": "改正個人情報保護法: 仮名加工情報と匿名加工情報の使い分けガイド",
                "content": "仮名加工情報: 他の情報と照合しない限り個人を特定できない状態に加工。内部利用のみ可、第三者提供不可。匿名加工情報: 個人を特定できないように不可逆加工。第三者提供可能だが加工基準が厳格。ビッグデータ分析には仮名加工情報が現実的。",
                "type": "decision",
                "tags": ["legal", "privacy", "data-protection"],
                "importance": 0.8,
            },
            {
                "summary": "電子契約の法的有効性: 電子署名法とe-文書法の要件整理",
                "content": "電子署名法第3条: 本人の意思に基づく電子署名は真正な成立を推定。事業者型電子署名（クラウドサイン、DocuSign等）は2条署名の要件を満たすかが論点。総務省・法務省Q&Aで固有性要件の充足を認める見解。",
                "type": "learning",
                "tags": ["legal", "e-signature", "contract"],
                "importance": 0.7,
            },
        ],
        "queries": [
            {
                "query": "仮名加工情報と匿名加工の違い",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "個人データの非識別化処理のやり方",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "データプライバシーの法的枠組み",
                "type": "concept",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "DocuSignで結んだ契約は裁判で有効？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "データ使いたいけど個人情報大丈夫？",
                "type": "vague",
                "expected_idx": 0,
                "difficulty": "hard",
            },
            {"query": "GDPR対応", "type": "cross", "expected_idx": None, "difficulty": "hard"},
            {
                "query": "知的財産権の国際登録",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "電子契約で取得した個人情報の仮名加工処理",
                "type": "multi-hop",
                "expected_idx": [0, 1],
                "difficulty": "hard",
            },
        ],
    },
    {
        "name": "agriculture",
        "display_name": "Agriculture",
        "summary": "Smart farming, crop management, agricultural policy",
        "memories": [
            {
                "summary": "スマート農業: ドローンによる可変施肥とNDVIによる生育診断",
                "content": "マルチスペクトルカメラ搭載ドローンでNDVI（正規化植生指数）マップを作成。圃場内の生育ムラを可視化し、可変施肥により肥料使用量を15-20%削減。DJI Agras T40やナイルワークスが国内主要機種。",
                "type": "note",
                "tags": ["agriculture", "drone", "precision-farming"],
                "importance": 0.8,
            },
            {
                "summary": "水田の水管理自動化: 自動給水栓とICT水位センサーの導入効果",
                "content": "paditch（パディッチ）やfarmo等のICT水管理システムにより、水田の水位をスマホで遠隔監視・制御。水管理労働時間を約80%削減。中干し期の適切な水位管理により収量・品質向上。",
                "type": "learning",
                "tags": ["agriculture", "water-management", "iot"],
                "importance": 0.7,
            },
        ],
        "queries": [
            {
                "query": "ドローン活用の精密農業",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "上空からの作物生育状況の把握",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "農業のIoT化",
                "type": "concept",
                "expected_idx": None,
                "difficulty": "medium",
            },
            {
                "query": "田んぼの水やりを自動化できる？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "農家の手間を減らすテクノロジー",
                "type": "vague",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "畜産のアニマルウェルフェア",
                "type": "cross",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "有機JAS認証の取得手順",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "ドローンで把握した生育ムラに基づく水管理最適化",
                "type": "multi-hop",
                "expected_idx": [0, 1],
                "difficulty": "hard",
            },
        ],
    },
    {
        "name": "marketing",
        "display_name": "Marketing & PR",
        "summary": "Digital marketing, PR strategy, brand management",
        "memories": [
            {
                "summary": "Cookie規制後のファーストパーティデータ戦略: CDP構築とコンテキスト広告への移行",
                "content": "Googleのサードパーティcookie廃止（Privacy Sandbox）に伴い、ファーストパーティデータの活用が急務。CDP（Treasure Data、Segment等）でデータ統合し、自社データベースで顧客セグメントを構築。コンテキスト広告（IAS、DoubleVerify）への投資が増加。",
                "type": "decision",
                "tags": ["marketing", "privacy", "first-party-data"],
                "importance": 0.8,
            },
            {
                "summary": "BtoB SaaSのPLG戦略: フリーミアムからの転換率最適化とプロダクト内オンボーディング",
                "content": "Product-Led Growth（PLG）ではフリーミアムの転換率2-5%が業界標準。Aha Momentまでの時間短縮が鍵。プロダクトツアー（Pendo、WalkMe）、In-app messaging、Usage-basedプライシングの組み合わせが効果的。",
                "type": "learning",
                "tags": ["marketing", "plg", "saas"],
                "importance": 0.7,
            },
        ],
        "queries": [
            {
                "query": "Cookie廃止後の広告戦略",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "サードパーティデータが使えなくなった後のマーケティング",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "デジタルマーケティングの転換期",
                "type": "concept",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "無料プランから有料プランへの転換率を上げるには？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "お客さんが課金してくれない",
                "type": "vague",
                "expected_idx": 1,
                "difficulty": "hard",
            },
            {
                "query": "インフルエンサーマーケティングのROI",
                "type": "cross",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "テレビCMの出稿プラン",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "Cookie規制下でPLG SaaSのユーザー獲得戦略",
                "type": "multi-hop",
                "expected_idx": [0, 1],
                "difficulty": "hard",
            },
        ],
    },
    {
        "name": "media-production",
        "display_name": "Media & Content Production",
        "summary": "Video production, music creation, streaming, content workflow",
        "memories": [
            {
                "summary": "バーチャルプロダクション: Unreal EngineとLEDウォール(ICVFX)による撮影ワークフロー",
                "content": "In-Camera VFX（ICVFX）はLEDウォールにUnreal Engineでリアルタイムレンダリングした背景を表示し撮影。カメラトラッキング（Mo-Sys StarTracker等）でパララックスを再現。グリーンバック合成と比べ、反射・ライティングが自然で俳優の演技にも好影響。",
                "type": "note",
                "tags": ["media", "virtual-production", "unreal-engine"],
                "importance": 0.8,
            },
            {
                "summary": "Dolby Atmos音楽制作: オブジェクトベースミキシングのDAWワークフロー",
                "content": "Dolby Atmos Rendererを使用し、Pro Tools/Logic Pro/Nuendoでオブジェクトベースミキシング。7.1.4ベッドにオブジェクトトラックを配置。Apple Music向けには.mp4/.movでマスタリング。バイノーラルダウンミックスの品質確認が重要。",
                "type": "learning",
                "tags": ["media", "dolby-atmos", "spatial-audio"],
                "importance": 0.7,
            },
        ],
        "queries": [
            {
                "query": "バーチャルプロダクションの撮影フロー",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "LEDスタジオでCG背景を使った撮影",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "映像制作のリアルタイム技術",
                "type": "concept",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "空間オーディオで音楽をミックスするには？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "映画っぽい映像を撮りたい",
                "type": "vague",
                "expected_idx": 0,
                "difficulty": "hard",
            },
            {
                "query": "YouTubeのサムネイル最適化",
                "type": "cross",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "ポッドキャストの収益化モデル",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "Unreal Engineで作った背景にAtmosサウンドを連動させる",
                "type": "multi-hop",
                "expected_idx": [0, 1],
                "difficulty": "hard",
            },
        ],
    },
    {
        "name": "investment",
        "display_name": "Investment & Asset Management",
        "summary": "Portfolio management, alternative investments, market analysis",
        "memories": [
            {
                "summary": "オルタナティブ投資の流動性リスク: プライベートデットとセカンダリー市場の動向",
                "content": "プライベートデット（直接融資）は金利上昇局面で注目。変動金利が主流でインフレヘッジに有効。ただし流動性が低く、ロックアップ期間2-5年。セカンダリー市場でのLP持分売却が流動性確保手段として拡大中。",
                "type": "note",
                "tags": ["investment", "alternatives", "private-debt"],
                "importance": 0.8,
            },
            {
                "summary": "ESG投資のグリーンウォッシング判定: SFDR分類とEUタクソノミー適合基準",
                "content": "EU SFDR: Article 6（ESG非考慮）、Article 8（ESG促進）、Article 9（サステナブル目的）の3分類。EUタクソノミーの6環境目標への実質的貢献とDNSH原則の充足が求められる。日本版はSSBJがIFRS S1/S2ベースで策定中。",
                "type": "decision",
                "tags": ["investment", "esg", "regulation"],
                "importance": 0.7,
            },
        ],
        "queries": [
            {
                "query": "プライベートデット投資のリスク",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "流動性の低い資産への投資戦略",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "サステナブルファイナンス",
                "type": "concept",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "ESGファンドが本当に環境に良いか見分ける方法は？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "最近流行りの投資先",
                "type": "vague",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "REIT市場の動向分析",
                "type": "cross",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "FXのスワップポイント計算",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "ESG準拠のプライベートデットファンドの流動性評価",
                "type": "multi-hop",
                "expected_idx": [0, 1],
                "difficulty": "hard",
            },
        ],
    },
    {
        "name": "energy",
        "display_name": "Energy & Utilities",
        "summary": "Renewable energy, power grid, energy storage, utility management",
        "memories": [
            {
                "summary": "系統用蓄電池のビジネスモデル: 需給調整市場と容量市場でのダブル収益化",
                "content": "系統用蓄電池は需給調整市場（一次〜三次調整力②）への応動で収益を得つつ、容量市場で固定収入を確保するダブルスタック戦略が主流。テスラMegapack（3.9MWh/ユニット）やBYD系蓄電池の導入が拡大。IRR 8-12%が目安。",
                "type": "note",
                "tags": ["energy", "battery-storage", "electricity-market"],
                "importance": 0.8,
            },
            {
                "summary": "ペロブスカイト太陽電池の商用化動向: 積水化学とエネコートの量産技術",
                "content": "ペロブスカイト太陽電池は軽量・フレキシブルで壁面設置可能。積水化学が2025年事業化目標、エネコートテクノロジーズがロールtoロール量産技術を開発中。変換効率は単接合で25%超を実験室レベルで達成。耐久性（10年以上）が商用化の最大課題。",
                "type": "learning",
                "tags": ["energy", "perovskite", "solar"],
                "importance": 0.7,
            },
        ],
        "queries": [
            {
                "query": "蓄電池の需給調整市場参入",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "大規模バッテリーで電力取引する方法",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "再生可能エネルギーの将来技術",
                "type": "concept",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "次世代のソーラーパネルってどんなやつ？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "電気代を下げたい",
                "type": "vague",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "水素燃料電池車の充填インフラ",
                "type": "cross",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "石油精製プロセスの最適化",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "ペロブスカイト電池で発電し蓄電池で需給調整に参入",
                "type": "multi-hop",
                "expected_idx": [0, 1],
                "difficulty": "hard",
            },
        ],
    },
    {
        "name": "logistics",
        "display_name": "Logistics & Supply Chain",
        "summary": "Freight transport, warehouse management, last-mile delivery",
        "memories": [
            {
                "summary": "2024年問題対応: トラックドライバー時間外規制と中継輸送・モーダルシフト戦略",
                "content": "2024年4月からトラックドライバーの時間外労働上限960時間/年が適用。長距離輸送対策として中継輸送拠点の整備、鉄道コンテナ・RORO船へのモーダルシフト、ダブル連結トラック（25m）の活用が進む。",
                "type": "decision",
                "tags": ["logistics", "2024-problem", "modal-shift"],
                "importance": 0.9,
            },
            {
                "summary": "自動配送ロボットの公道走行: 改正道路交通法の要件と実証事例",
                "content": "2023年4月の改正道交法で遠隔操作型小型車（配送ロボット）の公道走行が可能に。最高速度6km/h、遠隔監視者1名が要件。Panasonic、ZMP、Hakobot等が実証中。ラストワンマイル配送の人手不足対策として注目。",
                "type": "note",
                "tags": ["logistics", "delivery-robot", "last-mile"],
                "importance": 0.7,
            },
        ],
        "queries": [
            {
                "query": "2024年問題のトラック輸送対策",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "長距離ドライバーの労働時間規制への対応策",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "物流の自動化と省人化",
                "type": "concept",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "ロボットが荷物届けてくれるのって実現してる？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "荷物届くの遅くなるって聞いた",
                "type": "vague",
                "expected_idx": 0,
                "difficulty": "hard",
            },
            {
                "query": "倉庫のピッキングロボット",
                "type": "cross",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "国際海上コンテナの運賃指数",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "2024年問題で増える中継拠点にロボット配送を組み合わせる",
                "type": "multi-hop",
                "expected_idx": [0, 1],
                "difficulty": "hard",
            },
        ],
    },
    {
        "name": "manufacturing",
        "display_name": "Manufacturing & Industry 4.0",
        "summary": "Smart factory, quality control, industrial IoT",
        "memories": [
            {
                "summary": "予知保全(PdM)の実装: 振動センサーデータと機械学習による異常検知パイプライン",
                "content": "加速度センサー（IFM、バンナー）で設備振動を常時モニタリング。FFTで周波数特徴を抽出し、Isolation ForestまたはAutoEncoderで異常スコアを算出。閾値超過でアラート発報。故障予測精度85%以上で保全コスト30%削減の事例あり。",
                "type": "note",
                "tags": ["manufacturing", "predictive-maintenance", "ml"],
                "importance": 0.8,
            },
            {
                "summary": "デジタルツイン活用: NVIDIA OmniverseとPLC連携によるライン最適化",
                "content": "NVIDIA Omniverseで工場レイアウトの3Dデジタルツインを構築。PLC（三菱iQ-R、シーメンスS7）からリアルタイムデータを取得し、シミュレーションでボトルネック分析。ライン変更前のバーチャル検証により立上げ期間を40%短縮。",
                "type": "learning",
                "tags": ["manufacturing", "digital-twin", "nvidia"],
                "importance": 0.7,
            },
        ],
        "queries": [
            {
                "query": "設備の予知保全システム構築",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "機械の故障を事前に予測するAI",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "スマートファクトリーの実現",
                "type": "concept",
                "expected_idx": None,
                "difficulty": "medium",
            },
            {
                "query": "工場ラインのシミュレーションってどうやるの？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "工場の機械が壊れるの何とかしたい",
                "type": "vague",
                "expected_idx": 0,
                "difficulty": "hard",
            },
            {
                "query": "品質検査の自動化",
                "type": "cross",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "半導体製造のクリーンルーム基準",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "予知保全データをデジタルツインに統合してライン最適化",
                "type": "multi-hop",
                "expected_idx": [0, 1],
                "difficulty": "hard",
            },
        ],
    },
    {
        "name": "real-estate",
        "display_name": "Real Estate & PropTech",
        "summary": "Property management, real estate tech, urban development",
        "memories": [
            {
                "summary": "不動産STOの法的枠組み: 電子記録移転権利と不動産特定共同事業法の関係",
                "content": "不動産STO（Security Token Offering）は不動産特定共同事業法の電子取引業務ガイドラインに基づく。ブロックチェーン上のセキュリティトークンで不動産持分を小口化（最低投資額10万円〜）。ProgmatやSecuritize Japanがプラットフォームを提供。",
                "type": "decision",
                "tags": ["real-estate", "sto", "blockchain"],
                "importance": 0.8,
            },
            {
                "summary": "築古マンションのEER(エネルギー効率比)改善: 省エネ改修の投資回収シミュレーション",
                "content": "築30年超マンションの断熱改修（外壁：外断熱工法、窓：内窓追加）でエネルギー消費を30-40%削減。工事費は戸あたり200-300万円、光熱費削減で15-20年で回収。ZEH-M基準達成で補助金（最大100万円/戸）活用可能。",
                "type": "note",
                "tags": ["real-estate", "energy-retrofit", "sustainability"],
                "importance": 0.7,
            },
        ],
        "queries": [
            {
                "query": "不動産STOの仕組みと法規制",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "不動産のトークン化と小口投資",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "不動産テックの革新",
                "type": "concept",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "古いマンションの省エネリフォームは元が取れる？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "マンション老朽化どうする",
                "type": "vague",
                "expected_idx": 1,
                "difficulty": "hard",
            },
            {
                "query": "空き家バンクの活用事例",
                "type": "cross",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "商業施設のテナントリーシング戦略",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "STO対象マンションの省エネ改修による資産価値向上",
                "type": "multi-hop",
                "expected_idx": [0, 1],
                "difficulty": "hard",
            },
        ],
    },
    {
        "name": "cybersecurity",
        "display_name": "Cybersecurity",
        "summary": "Threat detection, incident response, security architecture",
        "memories": [
            {
                "summary": "ゼロトラストアーキテクチャ: NIST SP 800-207に基づくPDP/PEP実装パターン",
                "content": "ゼロトラストの中核はPDP（Policy Decision Point）とPEP（Policy Enforcement Point）。PDP: ユーザー属性・デバイス状態・リスクスコアを評価。PEP: APIゲートウェイやリバースプロキシで実装。Zscaler ZPA、Cloudflare Access、Azure ADがPDP/PEPを統合的に提供。",
                "type": "note",
                "tags": ["cybersecurity", "zero-trust", "nist"],
                "importance": 0.9,
            },
            {
                "summary": "ランサムウェアインシデント対応: 初動72時間の対応フローと身代金支払いの法的リスク",
                "content": "初動: ネットワーク隔離→感染範囲特定→フォレンジック保全→CSIRT/警察通報。身代金支払いはOFAC制裁対象国への支払いリスク（米国外国資産管理局）。バックアップからの復旧が原則。3-2-1ルール（3コピー、2種類の媒体、1つはオフサイト）の事前準備が重要。",
                "type": "decision",
                "tags": ["cybersecurity", "ransomware", "incident-response"],
                "importance": 0.9,
            },
        ],
        "queries": [
            {
                "query": "ゼロトラストの実装方式",
                "type": "exact",
                "expected_idx": 0,
                "difficulty": "easy",
            },
            {
                "query": "社内ネットワークを信頼しないセキュリティモデル",
                "type": "synonym",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "企業のセキュリティ戦略の転換",
                "type": "concept",
                "expected_idx": 0,
                "difficulty": "medium",
            },
            {
                "query": "ランサムウェアに感染したらまず何をする？",
                "type": "question",
                "expected_idx": 1,
                "difficulty": "medium",
            },
            {
                "query": "会社のデータが暗号化されて身代金要求された",
                "type": "vague",
                "expected_idx": 1,
                "difficulty": "hard",
            },
            {
                "query": "WAFの導入と運用",
                "type": "cross",
                "expected_idx": None,
                "difficulty": "hard",
            },
            {
                "query": "IoTデバイスのファームウェア署名検証",
                "type": "negative",
                "expected_idx": None,
                "difficulty": "control",
            },
            {
                "query": "ゼロトラスト環境でランサムウェア被害を最小化する方法",
                "type": "multi-hop",
                "expected_idx": [0, 1],
                "difficulty": "hard",
            },
        ],
    },
    # --- 表記ゆれ・同音異義語・名寄せ ---
    {
        "name": "japanese-nlp",
        "display_name": "Japanese NLP (表記ゆれ・同音異義語)",
        "summary": "Japanese text normalization, homonyms, conjugation, writing variations",
        "memories": [
            {
                "summary": "鯖の味噌煮レシピ: 生サバの下処理と臭み取りのコツ",
                "content": "鯖（さば）の味噌煮は下処理が重要。熱湯をかけて臭みを取り、生姜と味噌で煮込む。サバは青魚の中でもDHA・EPAが豊富。スーパーで買える生さばフィレが手軽。冷凍サバでも可。",
                "type": "note",
                "tags": ["料理", "魚", "鯖", "サバ", "さば"],
                "importance": 0.6,
            },
            {
                "summary": "日本橋から箸の専門店「箸長」への行き方と端の席の予約方法",
                "content": "日本橋にある箸（はし）の専門店「箸長」は橋（はし）のたもとに位置する。店の端（はし）の席は予約必須。江戸時代から続く伝統工芸の箸を販売。日本橋の橋の欄干からすぐ。",
                "type": "note",
                "tags": ["日本橋", "箸", "橋", "端", "はし"],
                "importance": 0.6,
            },
            {
                "summary": "マラソン大会で走った記録: 走る前のストレッチと走り方のフォーム改善",
                "content": "先月のマラソン大会で3時間切りを達成。走る前の動的ストレッチが重要。走った後のクールダウンで乳酸を流す。走り方のフォーム改善として、着地をミッドフット走法に変更。走れる距離が伸びた。",
                "type": "learning",
                "tags": ["マラソン", "走る", "走った", "ランニング"],
                "importance": 0.6,
            },
            {
                "summary": "引越し業者の見積もり比較: 引っ越し費用を安くするコツと引越し時期",
                "content": "引越し（引っ越し）の繁忙期は3-4月。引っこしの見積もりは最低3社で比較。引越し費用は平日が安い。ひっこし業者によって梱包サービスの有無が異なる。引越の準備は1ヶ月前から。",
                "type": "note",
                "tags": ["引越し", "引っ越し", "引越", "ひっこし"],
                "importance": 0.5,
            },
            {
                "summary": "サーバーとサーバの違い: ITにおける長音符の表記揺れとJIS規則",
                "content": "JIS Z 8301では3音以上の場合に長音符を省略可（サーバ、コンピュータ）。マイクロソフトは2008年から長音符ありに統一（サーバー、コンピューター）。プリンタ/プリンター、ユーザ/ユーザーも同様。現在は長音符ありが主流。",
                "type": "learning",
                "tags": ["IT", "表記揺れ", "長音符", "JIS"],
                "importance": 0.7,
            },
            {
                "summary": "お問い合わせフォームの設計: お問合せ/お問合わせの表記統一ガイドライン",
                "content": "Webフォームの表記は「お問い合わせ」が最も一般的。「お問合せ」「お問合わせ」「お問い合せ」も使用される。SEO観点では「お問い合わせ」に統一推奨。フォームのバリデーションで全角/半角、ハイフン有無の正規化が必要。",
                "type": "note",
                "tags": ["Web", "フォーム", "表記揺れ", "お問い合わせ"],
                "importance": 0.5,
            },
            {
                "summary": "雨の日の飴配りイベント: 飴(あめ)と雨(あめ)の同音異義語にまつわる販促企画",
                "content": "雨の日限定で飴を無料配布するキャンペーンが話題に。「あめの日にアメをプレゼント」というキャッチコピー。雨（天候）と飴（菓子）の同音異義語を活用したマーケティング。SNSでバズり来店数30%増。",
                "type": "note",
                "tags": ["マーケティング", "雨", "飴", "あめ", "同音異義語"],
                "importance": 0.5,
            },
            {
                "summary": "蜘蛛の巣の除去と雲の観察: くもに関する住宅メンテナンスと気象知識",
                "content": "蜘蛛（くも）の巣は軒下や窓枠に多い。酢水スプレーで忌避可能。一方、雲（くも）の種類は10種に分類（巻雲、積乱雲等）。蜘蛛の巣が多い日は晴れの兆候という民間伝承も。クモの巣除去スプレーは月1回が目安。",
                "type": "note",
                "tags": ["住宅", "蜘蛛", "雲", "くも", "メンテナンス"],
                "importance": 0.5,
            },
            {
                "summary": "食べたラーメン店レビュー: 食べる前の期待と食べ方のマナー",
                "content": "先週食べた二郎系ラーメンのレビュー。食べる前にニンニク・ヤサイ・アブラのコールが必要。食べ方は麺から先に。食べられる量を把握してからコール。食べすぎ注意。味は濃厚で食べごたえあり。食べたい人は早めに並ぶべし。",
                "type": "note",
                "tags": ["グルメ", "ラーメン", "食べる", "食べた", "レビュー"],
                "importance": 0.4,
            },
            {
                "summary": "子供が描いた絵の展示会: 描く技法と書く文字のバランス指導",
                "content": "小学校の展示会で子供が描いた絵を展示。描く（えがく/かく）と書く（かく）の使い分けを指導。絵を描く際はクレヨンと水彩の併用が効果的。字を書く練習と絵をかく練習を交互に実施。画く（えがく）は文語的表現。",
                "type": "note",
                "tags": ["教育", "描く", "書く", "かく", "展示会"],
                "importance": 0.5,
            },
        ],
        "queries": [
            {"query": "サバの味噌煮の作り方", "type": "exact", "expected_idx": 0, "difficulty": "easy"},
            {"query": "さばの下処理方法", "type": "synonym", "expected_idx": 0, "difficulty": "medium"},
            {"query": "鯖料理", "type": "concept", "expected_idx": 0, "difficulty": "medium"},
            {"query": "はしの専門店に行きたい", "type": "vague", "expected_idx": 1, "difficulty": "hard"},
            {"query": "日本橋の橋のそばの店", "type": "synonym", "expected_idx": 1, "difficulty": "hard"},
            {"query": "走った後のケア方法", "type": "exact", "expected_idx": 2, "difficulty": "easy"},
            {"query": "ランニングのフォーム改善", "type": "synonym", "expected_idx": 2, "difficulty": "medium"},
            {"query": "走る前にやること", "type": "question", "expected_idx": 2, "difficulty": "medium"},
            {"query": "ひっこしの費用を抑えるには？", "type": "synonym", "expected_idx": 3, "difficulty": "medium"},
            {"query": "引っ越し 見積もり 比較", "type": "exact", "expected_idx": 3, "difficulty": "easy"},
            {"query": "引越の準備スケジュール", "type": "synonym", "expected_idx": 3, "difficulty": "medium"},
            {"query": "サーバとサーバーどっちが正しい？", "type": "question", "expected_idx": 4, "difficulty": "medium"},
            {"query": "IT用語の長音符ルール", "type": "concept", "expected_idx": 4, "difficulty": "medium"},
            {"query": "コンピュータとコンピューターの違い", "type": "synonym", "expected_idx": 4, "difficulty": "hard"},
            {"query": "お問合せフォームの作り方", "type": "synonym", "expected_idx": 5, "difficulty": "medium"},
            {"query": "問い合わせページの表記統一", "type": "synonym", "expected_idx": 5, "difficulty": "hard"},
            {"query": "あめの日のキャンペーン", "type": "vague", "expected_idx": 6, "difficulty": "hard"},
            {"query": "雨の日の販促イベント", "type": "synonym", "expected_idx": 6, "difficulty": "medium"},
            {"query": "くもの巣の掃除方法", "type": "synonym", "expected_idx": 7, "difficulty": "medium"},
            {"query": "蜘蛛対策スプレー", "type": "exact", "expected_idx": 7, "difficulty": "easy"},
            {"query": "食べたラーメンの感想", "type": "exact", "expected_idx": 8, "difficulty": "easy"},
            {"query": "二郎系の食べ方ルール", "type": "synonym", "expected_idx": 8, "difficulty": "medium"},
            {"query": "子供がかいた絵", "type": "vague", "expected_idx": 9, "difficulty": "hard"},
            {"query": "描くと書くの違い", "type": "question", "expected_idx": 9, "difficulty": "medium"},
            {"query": "魚のさば", "type": "cross", "expected_idx": 0, "difficulty": "hard"},
            {"query": "箸の使い方マナー", "type": "negative", "expected_idx": None, "difficulty": "control"},
        ],
    },
    # --- PC パーツ EC 検索 ---
    {
        "name": "pc-parts-ec",
        "display_name": "PC Parts E-Commerce",
        "summary": "PC components, GPU, CPU, RAM, SSD, product specs and pricing",
        "memories": [
            {
                "summary": "NVIDIA GeForce RTX 4090 24GB GDDR6X: 最高性能GPU、TDP 450W、補助電源16pin",
                "content": "RTX 4090はAda Lovelace世代のフラッグシップ。CUDA 16384コア、ブースト2520MHz、24GB GDDR6X（384bit）。消費電力450Wで16pinコネクタ必要。4K 144Hzゲーミングに最適。価格帯25-30万円。DLSS 3対応でフレーム生成可能。",
                "type": "note",
                "tags": ["GPU", "NVIDIA", "RTX4090", "ビデオカード", "グラボ"],
                "importance": 0.8,
            },
            {
                "summary": "AMD Ryzen 9 7950X: 16コア32スレッド、AM5ソケット、TDP 170W",
                "content": "Ryzen 9 7950XはZen4アーキテクチャのハイエンドCPU。ベース4.5GHz/ブースト5.7GHz、L3キャッシュ64MB。AM5ソケット（LGA1718）でDDR5対応。170W TDPで水冷推奨。マルチスレッド性能はCore i9-13900Kを上回る。価格帯7-9万円。",
                "type": "note",
                "tags": ["CPU", "AMD", "Ryzen", "AM5", "プロセッサ"],
                "importance": 0.8,
            },
            {
                "summary": "Crucial DDR5-5600 32GB×2枚キット: デスクトップ向けメモリ、XMP3.0対応",
                "content": "Crucial DDR5-5600 CT2K32G56C46U5はDDR5-5600 CL46の64GBキット（32GB×2）。Intel XMP3.0/AMD EXPO対応で簡単OC。オンダイECC搭載で安定性向上。DDR4非互換、DDR5スロット必須。価格帯2-3万円。動画編集・3DCGに十分な容量。",
                "type": "note",
                "tags": ["メモリ", "DDR5", "RAM", "Crucial"],
                "importance": 0.7,
            },
            {
                "summary": "Samsung 990 PRO 2TB NVMe SSD: PCIe 4.0、読込7450MB/s、PS5対応",
                "content": "Samsung 990 PROはPCIe 4.0x4 NVMe M.2 SSD。シーケンシャル読込7450MB/s、書込6900MB/s。V-NAND TLC、DRAMキャッシュ搭載。PS5増設ストレージとしても人気。2TBで価格帯2-2.5万円。耐久性TBW 1200TB。ヒートシンク別売。",
                "type": "note",
                "tags": ["SSD", "NVMe", "Samsung", "ストレージ", "M.2"],
                "importance": 0.7,
            },
            {
                "summary": "Corsair RM850x ATX3.0電源: 80PLUS Gold、フルモジュラー、12VHPWR対応",
                "content": "Corsair RM850x (2024)はATX3.0対応850W電源。80PLUS Gold認証、フルモジュラーケーブル。12VHPWR（12+4pin）ケーブル付属でRTX 40系に直結可能。ファンレスモード搭載で低負荷時無音。10年保証。価格帯1.5-2万円。",
                "type": "note",
                "tags": ["電源", "PSU", "ATX3.0", "Corsair", "80PLUS"],
                "importance": 0.6,
            },
            {
                "summary": "ASUS ROG STRIX B650E-F マザーボード: AM5、DDR5、PCIe 5.0 M.2スロット搭載",
                "content": "ASUS ROG STRIX B650E-FはAM5対応ATXマザーボード。DDR5メモリ対応、PCIe 5.0 x16スロット1本、PCIe 5.0 M.2スロット1本搭載。2.5GbE LAN、WiFi 6E、USB4対応。BIOS FlashbackでCPUなしBIOS更新可能。価格帯3-4万円。",
                "type": "note",
                "tags": ["マザーボード", "ASUS", "AM5", "B650E", "M/B"],
                "importance": 0.7,
            },
            {
                "summary": "Noctua NH-D15 CPUクーラー: 空冷最強、デュアルタワー、165mm高",
                "content": "Noctua NH-D15はデュアルタワー空冷クーラーの定番。140mmファン2基搭載で静音性と冷却性能を両立。TDP 250W級CPUまで対応。高さ165mmでケース互換性に注意。LGA1700/AM5両対応マウンティングキット付属。価格帯1.2-1.5万円。",
                "type": "note",
                "tags": ["CPUクーラー", "空冷", "Noctua", "冷却"],
                "importance": 0.6,
            },
            {
                "summary": "LG 27GP950-B 4K 144Hz IPS ゲーミングモニター: HDMI2.1、HDR600、Nano IPS",
                "content": "LG 27GP950-Bは27インチ4K（3840x2160）144Hz IPSパネル。HDMI 2.1対応でPS5の4K 120Hz出力に対応。DisplayHDR 600認証、DCI-P3 98%カバー。応答速度1ms GtG。DisplayPort 1.4 + HDMI 2.1×2。USB Type-Cはなし。価格帯7-9万円。",
                "type": "note",
                "tags": ["モニター", "4K", "144Hz", "LG", "ゲーミング"],
                "importance": 0.7,
            },
            {
                "summary": "Fractal Design North ケース: 木目パネル、ATX対応、エアフロー重視設計",
                "content": "Fractal Design NorthはATX対応ミドルタワーケース。前面に天然ウォールナットの木目メッシュパネルでエアフロー確保。140mmファン2基付属。GPU最大355mm、CPUクーラー最大165mm。USB-C前面ポートあり。価格帯1.5-2万円。北欧デザインで人気。",
                "type": "note",
                "tags": ["PCケース", "Fractal Design", "ATX", "エアフロー"],
                "importance": 0.5,
            },
        ],
        "queries": [
            {"query": "RTX 4090 スペック", "type": "exact", "expected_idx": 0, "difficulty": "easy"},
            {"query": "4K 144Hzで遊べるグラボ", "type": "question", "expected_idx": 0, "difficulty": "medium"},
            {"query": "ビデオカード 最高性能", "type": "concept", "expected_idx": 0, "difficulty": "medium"},
            {"query": "グラフィックボード NVIDIA ハイエンド", "type": "synonym", "expected_idx": 0, "difficulty": "medium"},
            {"query": "Ryzen 7950X ベンチマーク性能", "type": "exact", "expected_idx": 1, "difficulty": "easy"},
            {"query": "AM5で一番いいCPU", "type": "question", "expected_idx": 1, "difficulty": "medium"},
            {"query": "動画編集に十分なメモリ容量", "type": "question", "expected_idx": 2, "difficulty": "medium"},
            {"query": "DDR5 64GB キット 価格", "type": "exact", "expected_idx": 2, "difficulty": "easy"},
            {"query": "PS5に増設できるSSD", "type": "question", "expected_idx": 3, "difficulty": "medium"},
            {"query": "えむどっとつー 高速ストレージ", "type": "vague", "expected_idx": 3, "difficulty": "hard"},
            {"query": "RTX4090に使える電源", "type": "question", "expected_idx": 4, "difficulty": "medium"},
            {"query": "12VHPWR 電源ユニット おすすめ", "type": "synonym", "expected_idx": 4, "difficulty": "medium"},
            {"query": "AM5対応のマザボでWiFiついてるやつ", "type": "vague", "expected_idx": 5, "difficulty": "hard"},
            {"query": "B650E マザーボード PCIe5.0", "type": "exact", "expected_idx": 5, "difficulty": "easy"},
            {"query": "空冷で最強のCPUファン", "type": "question", "expected_idx": 6, "difficulty": "medium"},
            {"query": "Noctua クーラー 高さ", "type": "exact", "expected_idx": 6, "difficulty": "easy"},
            {"query": "HDMI2.1対応の4Kモニター", "type": "question", "expected_idx": 7, "difficulty": "medium"},
            {"query": "PS5で4K120fps出せるディスプレイ", "type": "synonym", "expected_idx": 7, "difficulty": "medium"},
            {"query": "おしゃれな自作PCケース", "type": "vague", "expected_idx": 8, "difficulty": "hard"},
            {"query": "Fractal Design North 仕様", "type": "exact", "expected_idx": 8, "difficulty": "easy"},
            {"query": "自作PC パーツ一式 見積もり", "type": "concept", "expected_idx": None, "difficulty": "hard"},
            {"query": "Intel Core i9 13900K", "type": "negative", "expected_idx": None, "difficulty": "control"},
            {"query": "RTX4090に必要な電源容量とコネクタ", "type": "multi-hop", "expected_idx": [0, 4], "difficulty": "hard"},
            {"query": "Ryzen 7950XとDDR5メモリの互換性", "type": "multi-hop", "expected_idx": [1, 2], "difficulty": "hard"},
            {"query": "NH-D15がNorthケースに入るか", "type": "multi-hop", "expected_idx": [6, 8], "difficulty": "hard"},
            {"query": "4K144Hzモニターに必要なGPUとケーブル規格", "type": "multi-hop", "expected_idx": [0, 7], "difficulty": "hard"},
        ],
    },
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RecallResult:
    query: str
    query_type: str
    difficulty: str
    expected_idx: int | list[int] | None
    top_score: float
    top_summary: str
    all_scores: list[float]
    num_results: int
    latency_ms: float
    hit: bool  # expected memory found in results
    hit_rank: int | None  # rank of expected memory (1-indexed), None if not found


@dataclass
class DomainResult:
    domain: str
    embedding_model: str
    context_id: str
    remember_latency_ms: float
    recall_results: list[RecallResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(results: list[DomainResult]) -> dict:
    """Compute aggregate metrics from all results."""
    all_recalls: list[RecallResult] = []
    for dr in results:
        all_recalls.extend(dr.recall_results)

    if not all_recalls:
        return {}

    # Overall
    scored = [r for r in all_recalls if r.expected_idx is not None]
    neg = [r for r in all_recalls if r.expected_idx is None]

    top1_hits = sum(1 for r in scored if r.hit and r.hit_rank == 1)
    top3_hits = sum(1 for r in scored if r.hit and r.hit_rank is not None and r.hit_rank <= 3)

    # MRR
    mrr_sum = sum((1.0 / r.hit_rank) for r in scored if r.hit and r.hit_rank)

    latencies = [r.latency_ms for r in all_recalls]
    latencies_sorted = sorted(latencies)
    p50_idx = len(latencies_sorted) // 2
    p95_idx = int(len(latencies_sorted) * 0.95)

    # False positive: negative queries where top score > 0.8
    fp = sum(1 for r in neg if r.top_score > 0.8) if neg else 0

    # By query type
    by_type: dict[str, dict] = {}
    for qt in [
        "exact",
        "synonym",
        "concept",
        "question",
        "vague",
        "cross",
        "negative",
        "multi-hop",
    ]:
        type_results = [r for r in all_recalls if r.query_type == qt]
        type_scored = [r for r in type_results if r.expected_idx is not None]
        if type_results:
            by_type[qt] = {
                "count": len(type_results),
                "top1_acc": sum(1 for r in type_scored if r.hit and r.hit_rank == 1)
                / max(len(type_scored), 1),
                "avg_score": sum(r.top_score for r in type_results) / len(type_results),
                "avg_latency": sum(r.latency_ms for r in type_results) / len(type_results),
            }

    # By domain
    by_domain: dict[str, dict] = {}
    for dr in results:
        scored_d = [r for r in dr.recall_results if r.expected_idx is not None]
        by_domain[dr.domain] = {
            "top1_acc": sum(1 for r in scored_d if r.hit and r.hit_rank == 1)
            / max(len(scored_d), 1),
            "avg_score": sum(r.top_score for r in dr.recall_results)
            / max(len(dr.recall_results), 1),
            "avg_latency": sum(r.latency_ms for r in dr.recall_results)
            / max(len(dr.recall_results), 1),
            "remember_ms": dr.remember_latency_ms,
        }

    return {
        "total_queries": len(all_recalls),
        "scored_queries": len(scored),
        "top1_accuracy": top1_hits / max(len(scored), 1),
        "top3_accuracy": top3_hits / max(len(scored), 1),
        "mrr": mrr_sum / max(len(scored), 1),
        "avg_score": sum(r.top_score for r in all_recalls) / len(all_recalls),
        "score_stddev": _stddev([r.top_score for r in all_recalls]),
        "latency_p50": latencies_sorted[p50_idx] if latencies_sorted else 0,
        "latency_p95": latencies_sorted[p95_idx] if latencies_sorted else 0,
        "latency_avg": sum(latencies) / len(latencies),
        "false_positive_rate": fp / max(len(neg), 1),
        "by_type": by_type,
        "by_domain": by_domain,
    }


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


# ---------------------------------------------------------------------------
# Rich terminal output
# ---------------------------------------------------------------------------


def print_live_result(rr: RecallResult) -> None:
    icon = "[green]HIT[/]" if rr.hit else "[red]MISS[/]"
    rank_str = f"@{rr.hit_rank}" if rr.hit_rank else ""
    console.print(
        f"  {icon} [{rr.query_type:<9}] [{rr.difficulty:<7}] "
        f"score={rr.top_score:.3f} {rank_str:>3} "
        f"({rr.latency_ms:.0f}ms) {rr.query[:45]}"
    )


def print_summary_tables(
    metrics_by_model: dict[str, dict],
) -> None:
    """Print Rich summary tables."""

    # --- Model comparison ---
    t = Table(title="Model Comparison", show_lines=True)
    t.add_column("Metric", style="bold")
    for model in metrics_by_model:
        t.add_column(model, justify="right")

    rows = [
        ("Top-1 Accuracy", lambda m: f"{m['top1_accuracy']:.1%}"),
        ("Top-3 Accuracy", lambda m: f"{m['top3_accuracy']:.1%}"),
        ("MRR", lambda m: f"{m['mrr']:.3f}"),
        ("Avg Score", lambda m: f"{m['avg_score']:.3f}"),
        ("Score StdDev", lambda m: f"{m['score_stddev']:.3f}"),
        ("Latency p50", lambda m: f"{m['latency_p50']:.0f}ms"),
        ("Latency p95", lambda m: f"{m['latency_p95']:.0f}ms"),
        ("FP Rate (neg queries)", lambda m: f"{m['false_positive_rate']:.1%}"),
    ]
    for label, fn in rows:
        vals = [fn(metrics_by_model[model]) for model in metrics_by_model]
        t.add_row(label, *vals)
    console.print(t)

    # --- By query type ---
    t2 = Table(title="Accuracy by Query Type", show_lines=True)
    t2.add_column("Query Type", style="bold")
    for model in metrics_by_model:
        t2.add_column(f"{model}\nTop-1 Acc", justify="right")
        t2.add_column(f"{model}\nAvg Score", justify="right")

    all_types = [
        "exact",
        "synonym",
        "concept",
        "question",
        "vague",
        "cross",
        "negative",
        "multi-hop",
    ]
    for qt in all_types:
        row = [qt]
        for model in metrics_by_model:
            bt = metrics_by_model[model].get("by_type", {}).get(qt, {})
            row.append(f"{bt.get('top1_acc', 0):.0%}" if bt else "-")
            row.append(f"{bt.get('avg_score', 0):.3f}" if bt else "-")
        t2.add_row(*row)
    console.print(t2)

    # --- By domain (first model only for readability, full data in MD) ---
    first_model = next(iter(metrics_by_model))
    t3 = Table(title=f"Domain Results ({first_model})", show_lines=True)
    t3.add_column("Domain", style="bold")
    t3.add_column("Top-1 Acc", justify="right")
    t3.add_column("Avg Score", justify="right")
    t3.add_column("Avg Recall ms", justify="right")
    t3.add_column("Remember ms", justify="right")

    for domain, dm in metrics_by_model[first_model].get("by_domain", {}).items():
        t3.add_row(
            domain,
            f"{dm['top1_acc']:.0%}",
            f"{dm['avg_score']:.3f}",
            f"{dm['avg_latency']:.0f}",
            f"{dm['remember_ms']:.0f}",
        )
    console.print(t3)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def generate_markdown(
    metrics_by_model: dict[str, dict],
    all_results: dict[str, list[DomainResult]],
    embedding_info: list[dict],
) -> str:
    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Kagura Memory Embedding Benchmark Report",
        "",
        f"> Generated: {now}",
        "",
        "## Setup",
        "",
        "| Model | Provider | Dimensions | Status |",
        "|-------|----------|-----------|--------|",
    ]
    for m in embedding_info:
        lines.append(
            f"| {m['name']} | {m['provider']} | {m['dimensions']} | {'Available' if m['available'] else 'N/A'} |"
        )

    lines += [
        "",
        f"- **Domains**: {len(DOMAINS)}",
        "- **Query types**: exact, synonym, concept, question, vague, cross, negative, multi-hop",
        "- **Queries per domain**: 8",
        "",
        "## Model Comparison",
        "",
        "| Metric | " + " | ".join(metrics_by_model.keys()) + " |",
        "|--------|" + "|".join(["--------" for _ in metrics_by_model]) + "|",
    ]

    metric_rows = [
        ("Top-1 Accuracy", lambda m: f"{m['top1_accuracy']:.1%}"),
        ("Top-3 Accuracy", lambda m: f"{m['top3_accuracy']:.1%}"),
        ("MRR", lambda m: f"{m['mrr']:.3f}"),
        ("Avg Score", lambda m: f"{m['avg_score']:.3f}"),
        ("Score StdDev", lambda m: f"{m['score_stddev']:.3f}"),
        ("Latency p50", lambda m: f"{m['latency_p50']:.0f}ms"),
        ("Latency p95", lambda m: f"{m['latency_p95']:.0f}ms"),
        ("False Positive Rate", lambda m: f"{m['false_positive_rate']:.1%}"),
    ]
    for label, fn in metric_rows:
        vals = " | ".join(fn(metrics_by_model[model]) for model in metrics_by_model)
        lines.append(f"| {label} | {vals} |")

    # By query type
    lines += [
        "",
        "## Accuracy by Query Type",
        "",
        "| Query Type | "
        + " | ".join(f"{m} Top-1" for m in metrics_by_model)
        + " | "
        + " | ".join(f"{m} Score" for m in metrics_by_model)
        + " |",
        "|------------|" + "|".join(["--------" for _ in range(len(metrics_by_model) * 2)]) + "|",
    ]
    for qt in [
        "exact",
        "synonym",
        "concept",
        "question",
        "vague",
        "cross",
        "negative",
        "multi-hop",
    ]:
        row_parts = [qt]
        for model in metrics_by_model:
            bt = metrics_by_model[model].get("by_type", {}).get(qt, {})
            row_parts.append(f"{bt.get('top1_acc', 0):.0%}" if bt else "-")
        for model in metrics_by_model:
            bt = metrics_by_model[model].get("by_type", {}).get(qt, {})
            row_parts.append(f"{bt.get('avg_score', 0):.3f}" if bt else "-")
        lines.append("| " + " | ".join(row_parts) + " |")

    # By domain — full table for all models
    lines += ["", "## Domain Results", ""]
    for model in metrics_by_model:
        lines += [
            f"### {model}",
            "",
            "| Domain | Top-1 Acc | Avg Score | Recall ms | Remember ms |",
            "|--------|----------|----------|----------|------------|",
        ]
        bd = metrics_by_model[model].get("by_domain", {})
        for domain, dm in bd.items():
            lines.append(
                f"| {domain} | {dm['top1_acc']:.0%} | {dm['avg_score']:.3f} | {dm['avg_latency']:.0f} | {dm['remember_ms']:.0f} |"
            )
        lines.append("")

    # Detailed per-query results (collapsible)
    lines += ["## Detailed Query Results", ""]
    for model, dr_list in all_results.items():
        lines += [f"### {model}", ""]
        for dr in dr_list:
            lines += [
                f"<details><summary>{dr.domain}</summary>",
                "",
                "| Type | Difficulty | Score | Rank | Hit | Query |",
                "|------|-----------|-------|------|-----|-------|",
            ]
            for rr in dr.recall_results:
                hit_str = "HIT" if rr.hit else "MISS"
                rank_str = str(rr.hit_rank) if rr.hit_rank else "-"
                lines.append(
                    f"| {rr.query_type} | {rr.difficulty} | {rr.top_score:.3f} | {rank_str} | {hit_str} | {rr.query} |"
                )
            lines += ["", "</details>", ""]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _check_hit(
    results: list[dict],
    domain_memories: list[dict],
    expected_idx: int | list[int] | None,
) -> tuple[bool, int | None]:
    """Check if expected memory is in recall results. Returns (hit, rank)."""
    if expected_idx is None:
        return False, None  # negative/cross queries — no expected answer

    expected_indices = expected_idx if isinstance(expected_idx, list) else [expected_idx]
    expected_summaries = [domain_memories[i]["summary"] for i in expected_indices]

    for rank, result in enumerate(results, 1):
        result_summary = result.get("summary", "")
        for es in expected_summaries:
            # Partial match: first 30 chars of expected summary
            if es[:30] in result_summary or result_summary[:30] in es:
                return True, rank
    return False, None


async def run_benchmark(cleanup: bool = False) -> None:
    config = load_config()
    api_key = config.get("api_key", "")
    mcp_url = config.get("mcp_url", "")

    if not api_key or not mcp_url:
        console.print("[red]Error: .kagura.json with api_key and mcp_url required[/]")
        sys.exit(1)

    async with KaguraClient(api_key=api_key, mcp_url=mcp_url) as client:
        # Embedding models info
        console.rule("Embedding Models")
        models_resp = await client.list_embedding_models()
        embedding_info = []
        for m in models_resp.models:
            status = "[green]available[/]" if m.available else "[red]unavailable[/]"
            console.print(f"  {m.name} ({m.provider}, {m.dimensions}dim) {status}")
            embedding_info.append(
                {
                    "name": m.name,
                    "provider": m.provider,
                    "dimensions": m.dimensions,
                    "available": m.available,
                }
            )
        console.print(f"  Default: {models_resp.default_model}")

        if cleanup:
            await _cleanup(client)
            return

        # Build context index
        existing = await client.list_contexts()
        ctx_index: dict[str, str] = {c["name"]: c["id"] for c in existing.get("contexts", [])}

        all_results: dict[str, list[DomainResult]] = {}

        for emb_model in EMBEDDING_MODELS:
            console.rule(f"[bold]{emb_model}[/bold]")
            model_results: list[DomainResult] = []

            for domain in DOMAINS:
                domain_name = domain["name"]
                ctx_name = f"bench-{emb_model.replace(':', '-').replace('.', '')}-{domain_name}"[
                    :50
                ]

                console.print(f"\n[bold cyan]{domain['display_name']}[/] ({emb_model})")

                # Create or reuse context
                if ctx_name in ctx_index:
                    context_id = ctx_index[ctx_name]
                    console.print(f"  Context: {context_id[:8]}... (existing)")
                else:
                    try:
                        ctx = await client.create_context(
                            name=ctx_name,
                            display_name=f"[Bench] {domain['display_name']}",
                            summary=domain["summary"],
                            is_private=True,
                            embedding_model=emb_model,
                        )
                        context_id = ctx.get("context_id", "")
                        if not context_id:
                            console.print(f"  [red]SKIP (no context_id): {ctx}[/]")
                            continue
                        ctx_index[ctx_name] = context_id
                        console.print(f"  Context: {context_id[:8]}... (created)")
                    except Exception as e:
                        console.print(f"  [red]SKIP: {e}[/]")
                        continue

                # Remember (skip if already populated)
                probe = await client.recall(context_id=context_id, query="test", k=1)
                if probe.get("results"):
                    remember_ms = 0.0
                    console.print("  Memories: already populated")
                else:
                    t0 = time.monotonic()
                    for mem in domain["memories"]:
                        await client.remember(
                            context_id=context_id,
                            summary=mem["summary"],
                            content=mem["content"],
                            type=mem["type"],
                            tags=mem["tags"],
                            importance=mem["importance"],
                        )
                    remember_ms = (time.monotonic() - t0) * 1000
                    console.print(
                        f"  Remembered {len(domain['memories'])} items ({remember_ms:.0f}ms)"
                    )
                    await asyncio.sleep(1.0)

                # Recall queries
                dr = DomainResult(
                    domain=domain_name,
                    embedding_model=emb_model,
                    context_id=context_id,
                    remember_latency_ms=remember_ms,
                )

                for q in domain["queries"]:
                    t0 = time.monotonic()
                    results = await client.recall(context_id=context_id, query=q["query"], k=5)
                    latency = (time.monotonic() - t0) * 1000
                    hits = results.get("results", [])
                    top = hits[0] if hits else None

                    hit, rank = _check_hit(hits, domain["memories"], q["expected_idx"])

                    rr = RecallResult(
                        query=q["query"],
                        query_type=q["type"],
                        difficulty=q["difficulty"],
                        expected_idx=q["expected_idx"],
                        top_score=top["score"] if top else 0.0,
                        top_summary=(top["summary"][:60] + "...") if top else "NO RESULTS",
                        all_scores=[h["score"] for h in hits],
                        num_results=len(hits),
                        latency_ms=latency,
                        hit=hit,
                        hit_rank=rank,
                    )
                    dr.recall_results.append(rr)
                    print_live_result(rr)

                model_results.append(dr)

            all_results[emb_model] = model_results

        # Compute metrics
        metrics_by_model = {}
        for model, drs in all_results.items():
            metrics_by_model[model] = compute_metrics(drs)

        # Terminal summary
        console.rule("[bold green]Results[/bold green]")
        print_summary_tables(metrics_by_model)

        # Markdown report
        md = generate_markdown(metrics_by_model, all_results, embedding_info)
        md_path = "benchmark_report.md"
        with open(md_path, "w") as f:
            f.write(md)
        console.print(f"\n[green]Report saved to {md_path}[/]")

        # JSON raw data
        json_data = {
            model: [
                {
                    "domain": dr.domain,
                    "embedding_model": dr.embedding_model,
                    "context_id": dr.context_id,
                    "remember_latency_ms": dr.remember_latency_ms,
                    "direct_recalls": [
                        {
                            "query": rr.query,
                            "type": rr.query_type,
                            "difficulty": rr.difficulty,
                            "top_score": rr.top_score,
                            "hit": rr.hit,
                            "hit_rank": rr.hit_rank,
                            "latency_ms": rr.latency_ms,
                            "num_results": rr.num_results,
                            "all_scores": rr.all_scores,
                        }
                        for rr in dr.recall_results
                    ],
                }
                for dr in drs
            ]
            for model, drs in all_results.items()
        }
        json_path = "benchmark_results.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        console.print(f"[green]Raw data saved to {json_path}[/]")


async def _cleanup(client: KaguraClient) -> None:
    console.print("[bold]Cleaning up benchmark contexts...[/]")
    contexts = await client.list_contexts()
    count = 0
    for ctx in contexts.get("contexts", []):
        if ctx["name"].startswith("bench-"):
            console.print(f"  {ctx['name']} ({ctx['id'][:8]}...) — memories cleared")
            try:
                await client._call_tool("forget", {"context_id": ctx["id"], "memory_id": "all"})
            except Exception:
                pass
            count += 1
    console.print(f"  {count} benchmark contexts found (context deletion requires admin API)")


if __name__ == "__main__":
    do_cleanup = "--cleanup" in sys.argv
    asyncio.run(run_benchmark(cleanup=do_cleanup))
